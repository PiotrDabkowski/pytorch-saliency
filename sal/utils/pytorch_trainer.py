
import time
import sys
import numpy as np
import os
import cPickle
import torch.nn
import torch.utils.data as torch_data
import torch.optim as torch_optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.modules.loss as losses

from torch.autograd import Variable
import threading
import pycat


INFO_TEMPLATE = '\033[38;5;2mINFO: %s\033[0m\n'
WARN_TEMPLATE = '\033[38;5;1mWARNING: %s\033[0m\n'

assert torch.cuda.is_available(), 'CUDA must be available'

class PTStore:
    def __init__(self):
        self.__dict__['vars'] = {}

    def __call__(self, **kwargs):
        assert len(kwargs)==1, "You must specify just 1 variable to add"
        key, value = kwargs.items()[0]
        setattr(self, key, value)
        return value

    def __setattr__(self, key, value):
        self.__dict__['vars'][key] = value

    def __getattr__(self, key):
        if key=='_vars':
            return self.__dict__['vars']
        if key not in self.__dict__['vars']:
            raise KeyError('Key %s was not found in the pt_store! Forgot to add it?' % key)
        return self.__dict__['vars'][key]

    def __getitem__(self, key):
        if key not in self.__dict__['vars']:
            raise KeyError('Key %s was not found in the pt_store! Forgot to add it?' % key)
        cand = self.__dict__['vars'][key]
        return to_numpy(cand)

    def clear(self):
        self.__dict__['vars'].clear()



PT = PTStore()
BATCHES_DONE_INFO = '{batches_done}/{batches_per_epoch}'
TIME_INFO = 'time: {comp_time:.3f} - data: {data_time:.3f} - ETA: {eta:.0f}'
SPEED_INFO = 'e/s: {examples_per_sec:.1f}'


def to_numpy(cand):
    if isinstance(cand, Variable):
        return cand.data.cpu().numpy()
    elif isinstance(cand, torch._TensorBase):
        return cand.cpu().numpy()
    elif isinstance(cand, (list, tuple)):
        return map(to_numpy, cand)
    elif isinstance(cand, np.ndarray):
        return cand
    else:
        return np.array([cand])

def to_number(x):
    if isinstance(x, (int, long, float)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x[0]
    return x.data[0]



def smoothing_dict_update(main, update, smooth_coef):
    for k, v in update.items():
        if main.get(k) is None:
            main[k] = v
        else:
            main[k] = smooth_coef*main[k] + (1.-smooth_coef)*v
    return main


class NiceTrainer:
    def __init__(self,
                 forward_step,  # forward step takes the output of the transform_inputs
                 train_dts,
                 optimizer,
                 pt_store=PT,
                 transform_inputs=lambda batch, trainer: batch,

                 printable_vars=(),
                 events=(),
                 computed_variables=None,

                 loss_name='loss',

                 val_dts=None,
                 set_trainable=None,

                 modules=None,
                 save_every=None,
                 save_dir='ntsave',

                 info_string=(BATCHES_DONE_INFO, TIME_INFO, SPEED_INFO),
                 smooth_coef=0.95,
                 goodbye_after = 5,

                 lr_step_period=None,
                 lr_step_gamma=0.1,
                 ):
        '''

        '''
        self.forward_step = forward_step
        assert isinstance(train_dts, torch_data.DataLoader),  'train_dts must be an instance of torch.utils.data.DataLoader'
        self.train_dts = train_dts

        assert isinstance(optimizer, torch_optim.Optimizer), 'optimizer must be an instance of torch.optim.Optimizer'
        self.optimizer = optimizer
        assert isinstance(pt_store, PTStore), 'pt_store must be an instance of PTStore'
        self.pt_store = pt_store
        self.transform_inputs = transform_inputs


        self.printable_vars = list(printable_vars)
        self.events = list(events) if events else []
        assert all(map(lambda x: isinstance(x, BaseEvent), self.events)), 'All events must be instances of the BaseEvent!'
        self.computed_variables = computed_variables if computed_variables is not None else {}

        self.loss_name = loss_name


        assert val_dts is None or isinstance(val_dts, torch_data.DataLoader),  'val_dts must be an instance of torch.utils.data.DataLoader or None'
        self.val_dts = val_dts
        if modules is not None:
            if not hasattr(modules, '__iter__'):
                modules = [modules]
        assert modules is None or all(map(lambda x: isinstance(x, torch.nn.Module), modules)), 'The list of modules can only contain instances of torch.nn.Module'
        self.modules = modules
        if set_trainable is None and self.modules is not None:
            def set_trainable(is_training):
                for m in self.modules:
                    m.train(is_training)

        self.set_trainable = set_trainable


        # todo implement save/restore functionality!
        self.save_every = save_every
        self.save_dir = save_dir
        if self.save_every is not None:
            self._add_timed_save_event(save_every)

        self.smooth_coef = smooth_coef


        self._is_in_train_mode = None
        self.goodbye_after = goodbye_after

        self._extra_var_info_string = (info_string if isinstance(info_string, basestring) else ' - '.join(info_string)) + (
            (' - ' if self.printable_vars else '') + ' - '.join('%s: {%s:.4f}' % (e, e) for e in self.printable_vars))



        self.info_vars = dict(
            epochs_done=0,
            batches_done=0,
            total_batches_done=0,
            batch_size=0,
            total_examples_done=0,
            batches_per_sec=float('nan'),
            examples_per_sec=float('nan'),
            batches_per_epoch=float('nan'),
            eta=float('nan'),
            data_time=float('nan'),
            comp_time=float('nan'),
            is_training=None,
        )
        self._core_info_vars = set(self.info_vars.keys())

        if lr_step_period is not None:
            self._add_lr_scheduling_event(lr_step_period, lr_step_gamma)

    def _add_timed_save_event(self, period):
        @TimeEvent(period=period, first_at=period)
        def periodic_save(s):
            print
            print INFO_TEMPLATE % 'Performing a periodic save...'
            s.save()
        self.events.append(periodic_save)

    def _add_lr_scheduling_event(self, period, gamma):
        @TrainStepEvent()
        @EpochChangeEvent()
        def lr_step_event(s):
            if not hasattr(s, '_lr_sheduler') or s._lr_sheduler is None:
                s._lr_sheduler = StepLR(s.optimizer, period, gamma)
            s._lr_sheduler.step(epoch=s.info_vars['epochs_done'])
            for param_group in s.optimizer.param_groups:
                print INFO_TEMPLATE % ('LR ' + str(param_group['lr']))
        self.events.append(lr_step_event)




    def _main_loop(self, is_training, steps=None, allow_switch_mode=True):
        """Trains for 1 epoch if steps is None. Otherwise performs specified number of steps."""
        assert steps is None, 'Not supported yet!'  # todo allow continue and partial execution
        if not is_training:
            assert self.val_dts is not None, 'Validation dataset was not provided'
        if allow_switch_mode and self._is_in_train_mode != is_training:
            if self.set_trainable is not None:
                self.set_trainable(is_training=is_training)
                self._is_in_train_mode = is_training
            else:
                if is_training:
                    print WARN_TEMPLATE % "could not set the modules to the training mode because neither set_trainable nor modules were provided, assuming already in the training mode"
                    self._is_in_train_mode = True
                else:
                    raise ValueError("cannot set the modules to the eval mode because neither set_trainable nor modules were provided")

        dts = self.train_dts if is_training else self.val_dts
        dts_iter = iter(dts)
        smooth_burn_in = 0.8 * self.smooth_coef
        smooth_normal = self.smooth_coef

        smoothing_dict = dict(
            comp_time=None,
            data_time=None,
        )
        smoothing_dict.update(dict(zip(self.printable_vars, len(self.printable_vars)*[None])))


        batches_per_epoch = len(dts)
        batch_size = dts.batch_size
        steps_done_here = 0
        batches_done = 0  # todo allow continue!

        self.info_vars.update(dict(
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            batches_done=batches_done,
            is_training=is_training,
        ))



        last_hearbeat = [time.time()+5.]
        # def guard():
        #     while 1:
        #         time.sleep(1)
        #         last = last_hearbeat[0]
        #         if last is None:
        #             break
        #         if time.time()-last>self.goodbye_after:
        #             print 'We have not received any heartbeat update since last %d seconds. Time to say goodbye!' % self.goodbye_after
        #             os.system('kill %d' % os.getpid())
        #             break
        # t = threading.Thread(target=guard)
        # t.daemon = True
        # t.start()



        smoothed_printable_vars = {}

        t_fetch = time.time()
        for batch in dts_iter:
            last_hearbeat[0] = time.time()

            self.pt_store.clear()
            batch = self.transform_inputs(batch, self)
            self.pt_store.batch = batch

            torch.cuda.synchronize()
            t_start = time.time()
            # --------------------------- OPTIMIZATION STEP ------------------------------------
            if is_training:
                self.optimizer.zero_grad()

            self.forward_step(batch)
            loss = getattr(self.pt_store, self.loss_name)

            if is_training:
                loss.backward()
                self.optimizer.step()
            # ----------------------------------------------------------------------------------
            torch.cuda.synchronize()
            t_end = time.time()

            # call events
            for event in self.events:
                event(self)


            # smoothing coef should be relatively small at the start because the starting values are very unstable
            # todo use bias correction by dividing by (1-smooth^t)
            smooth_coef = smooth_normal if steps_done_here > 22 else smooth_burn_in

            # important to smooth computation times to have a nice estimates
            smoothing_update = dict(
                comp_time= t_end - t_start,
                data_time= t_start - t_fetch,
            )

            # calculate computed variables and add them to the pt_store
            for var_name, func in self.computed_variables.items():
                setattr(self.pt_store, var_name, func(self))

            # add ALL the printable variables to the smoother
            smoothing_update.update({e:to_number(self.pt_store[e]) for e in self.printable_vars if e not in self._core_info_vars})

            # perform smoother update
            smoothing_dict_update(smoothed_printable_vars, smoothing_update, smooth_coef)

            # calculate
            batches_per_sec = 1./ (smoothed_printable_vars['comp_time']+smoothed_printable_vars['data_time'])
            batches_done += 1
            steps_done_here += 1
            eta = (batches_per_epoch - batches_done) / batches_per_sec

            # update info vars after iteration step
            self.info_vars.update(dict(
                batches_done=batches_done,
                total_batches_done=self.info_vars['total_batches_done']+1,
                total_examples_done=self.info_vars['total_examples_done']+batch_size,
                batches_per_sec=batches_per_sec,
                examples_per_sec=batches_per_sec*batch_size,
                eta=eta,
            ))
            # info vars should also contain all the printable vars
            self.info_vars.update(smoothed_printable_vars)





            formatted_info_string = self._extra_var_info_string.format(**self.info_vars)
            sys.stdout.write('\r' + formatted_info_string)
            sys.stdout.flush()

            t_fetch = t_end
        else:
            if is_training:
                self.info_vars['epochs_done'] += 1

        # we have left the dangerous loop, quit the guarding process...
        last_hearbeat[0] = None
        sys.stdout.write('\n')
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def train(self, steps=None):
        if steps is None:
            print '_'*55
            print 'Epoch', self.info_vars['epochs_done']+1
        self._main_loop(is_training=True, steps=steps, allow_switch_mode=True)

    def validate(self, allow_switch_mode=False):
        old_info = self.info_vars.copy()
        print "Validation:"
        self._main_loop(is_training=False, steps=None, allow_switch_mode=allow_switch_mode)
        self.info_vars = old_info


    def _get_state(self):
        return dict(
            info_vars={k:v for k, v in self.info_vars.items() if k in self._core_info_vars},
            state_dicts=[m.state_dict() for m in self.modules],
            optimizer_state=self.optimizer.state_dict(),
        )

    def _set_state(self, state):
        self.info_vars = state['info_vars']
        self.optimizer.load_state_dict(state['optimizer_state'])
        if len(self.modules)!=len(state['state_dicts']):
            raise ValueError('The number of save dicts is different from number of models')
        for m, s in zip(self.modules, state['state_dicts']):
            m.load_state_dict(s)

    def save(self, step=1):
        if not self.modules:
            raise ValueError("nothing to save - the list of modules was not provided")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        torch.save(self._get_state(), os.path.join(self.save_dir, 'model-%d.ckpt'%step))


    def restore(self, step=1):
        if not self.modules:
            raise ValueError("nothing to load - the list of modules was not provided")
        p = os.path.join(self.save_dir, 'model-%d.ckpt' % step)
        if not os.path.exists(p):
            return
        self._set_state(torch.load(p))

    #
    #
    # def restore(self, allow_restore_crash=True, relaxed=False):
    #     """ If you set allow_restore_crash to True we will
    #     check whether automatic periodic save was made after standard save and if this is the case
    #     we will continue from periodic save."""
    #     assert self.saver is not None, 'You must specify saver if you want to use restore'
    #
    #     std_checkpoint = tf.train.get_checkpoint_state(self.save_dir)
    #     if std_checkpoint and std_checkpoint.model_checkpoint_path:
    #         step = int(std_checkpoint.model_checkpoint_path.split('-')[-1])
    #         std_nt = cPickle.load(open(os.path.join(self.save_dir, 'nice_trainer%d'%step), 'rb'))
    #     else:
    #         std_nt = None
    #
    #     if allow_restore_crash:
    #         # check periodic save folder
    #         periodic_check_dir = os.path.join(self.save_dir, 'periodic_check')
    #         periodic_checkpoint = tf.train.get_checkpoint_state(periodic_check_dir)
    #         if periodic_checkpoint and periodic_checkpoint.model_checkpoint_path:
    #             periodic_nt = cPickle.load(open(os.path.join(periodic_check_dir, 'nice_trainer%d' % 0), 'rb'))
    #         else:
    #             periodic_nt = None
    #         if periodic_nt is not None and (std_nt is None or std_nt['last_save_time'] < periodic_nt['last_save_time']):
    #             # restore from crash
    #             print 'Restoring from periodic save (maybe in the middle of the epoch). Training will be continued.'
    #             self._restore(periodic_checkpoint.model_checkpoint_path, relaxed)
    #             self._restore_nt(periodic_nt, continue_epoch=True)
    #             return
    #
    #     if std_checkpoint and std_checkpoint.model_checkpoint_path:
    #         print 'Loading model from', std_checkpoint.model_checkpoint_path
    #         self._restore(std_checkpoint.model_checkpoint_path, relaxed)
    #         self._restore_nt(std_nt)
    #         return
    #     else:
    #         print 'No saved models to restore from'
    #         return
    #
    # def _restore(self, path, relaxed):
    #     if not relaxed:
    #         self.saver.restore(self.sess, path)
    #     else:
    #         optimistic_restore(self.sess, path, var_list=self.saver._var_list)
    #
    # def _save_nt(self, save_path):
    #     nt = {
    #         'last_save_time': self._last_save_time,
    #         'save_num': self._save_num,
    #         'epoch': self._epoch,
    #         'measured_batches_per_sec': self.measured_batches_per_sec,
    #         'train_bm_state': self.bm_train.get_state(),
    #         'logs': self.logs,
    #     }
    #     cPickle.dump(nt, open(save_path, 'wb'))
    #
    #
    #
    # def _restore_nt(self, old_nt, continue_epoch=False):
    #     self._epoch = old_nt['epoch']
    #     self._save_num = old_nt['save_num']
    #     self.measured_batches_per_sec = old_nt['measured_batches_per_sec']
    #     self.logs = old_nt.get('logs', self.logs)
    #
    #     if continue_epoch:
    #         self._epoch -= 1
    #         # now make changes to the train batch manager...
    #         self.bm_train.continue_from_state(old_nt['train_bm_state'])



class BaseEvent:
    def __init__(self):
        self.func = None

    def __call__(self, func_or_trainer):
        if self.func is None:
            self.func = func_or_trainer
            return self
        if self.should_run(func_or_trainer):
            self.func(func_or_trainer)

    def should_run(self, trainer):
        raise NotImplementedError()


class StepEvent(BaseEvent):
    def should_run(self, trainer):
        return True

class ValStepEvent(BaseEvent):
    def should_run(self, trainer):
        return not trainer.info_vars['is_training']

class TrainStepEvent(BaseEvent):
    def should_run(self, trainer):
        return trainer.info_vars['is_training']


class EpochChangeEvent(BaseEvent):
    def __init__(self, call_on_first=True):
        BaseEvent.__init__(self)
        self.last_epoch = None
        self.call_on_first = call_on_first

    def should_run(self, trainer):
        current = trainer.info_vars['epochs_done']
        if self.last_epoch is None:
            self.last_epoch = current
            if self.call_on_first:
                return True
            else:
                return False
        if self.last_epoch!=current:
            self.last_epoch = current
            return True
        return False


class EveryNthEvent(BaseEvent):
    def __init__(self, every_n, required_remainder=0):
        BaseEvent.__init__(self)
        self.every_n = every_n
        self.required_remainder = required_remainder % every_n
        self.count = 0

    def should_run(self, trainer):
        self.count += 1
        return self.count%self.every_n == self.required_remainder



class TimeEvent(BaseEvent):
    def __init__(self, period, first_at=0):
        assert first_at <= period
        BaseEvent.__init__(self)
        self.period = period
        self.last = time.time()-(period-first_at)

    def should_run(self, trainer):
        t = time.time()
        if t>=self.last+self.period:
            self.last = t
            return True
        return False



def img_show_event(imgs_name, every_n_seconds=5, ith=0):
    @TimeEvent(period=every_n_seconds)
    def f(s):
        pycat.show(s.pt_store['imgs_name'][ith])
        print
    return f


def _caclulate_batch_top_n_hits(probs, labels, n, avg_preds_over):
    ''' probs is a probabilit matrix (BS, N) labels is a vector (BS,)  with every entry smaller int than N'''
    if n==1:
        return np.mean(np.argmax(probs, 1)==labels)
    else:
        # generic top n accuracy, generally fast enough...
        return np.mean((np.sum(np.cumsum(np.argsort(probs, axis=1)==np.expand_dims(labels, 1), 1), 1) <= n).astype(np.float32))


def accuracy_calc_op(logits_from='logits', labels_from='labels', top_n=1, avg_preds_over=1):
    '''
    Calculates top n accuracy.
    You can average predictions from a number of consecutive evaluations as specified by avg_preds_over.
    It is required that BATCH_SIZE is divisible by avg_preds_over and total number of distinct examples in one batch is
    BATCH_SIZE/avg_preds_over.

        Note: requires that extra var named 'probs' (BATCH_SIZE, NUM_CLASSES) with class probabilities is present in extra_vars.
              Also labels must be simply (BATCH_SIZE,) '''
    def acc_op(trainer):
        _logits = trainer.pt_store[logits_from]
        _labels = trainer.pt_store[labels_from]
        return _caclulate_batch_top_n_hits(_logits, _labels, top_n, avg_preds_over)
    return acc_op



def ev_batch_to_images_labels(func):
    def f(batch):
        _images, _labels = batch
        _images = PT(images=Variable(_images).cuda())
        _labels = PT(labels=Variable(_labels).cuda())
        return func(_images, _labels)
    return f


def simple_img_classifier_train(model, dts, batch_size=512, epochs=25, lr=0.1, lr_step=11, weight_decay=0.0001):
    train_dts = dts.get_train_dataset()
    val_dts = dts.get_val_dataset()

    p_c = torch.nn.DataParallel(model).cuda()
    criterion = losses.CrossEntropyLoss()
    optim = torch_optim.SGD(model.parameters(), lr, 0.9, weight_decay=weight_decay, nesterov=True)

    @ev_batch_to_images_labels
    def ev(_images, _labels):
        out = p_c(_images)
        logits = model.out_to_logits(out)
        PT(logits=logits)
        loss = PT(loss=criterion(logits, _labels))

    nt = NiceTrainer(ev, dts.get_loader(train_dts, batch_size),
                     optim, printable_vars=['loss', 'acc'],
                     computed_variables={'acc': accuracy_calc_op()},
                     lr_step_period=lr_step,
                     val_dts=dts.get_loader(val_dts, batch_size))

    for e in xrange(epochs):
        nt.train()
    nt.validate()


import cv2
from Queue import Queue
import time
import numpy as np
import threading
from PIL import Image
from sal.datasets.imagenet_dataset import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
import wx

TO_SHOW = 737
CONFIDENCE = 4
POLL_DELAY = 0.01


def numpy_to_wx(image):
    height, width, c = image.shape
    image = np.flip(image, 2)
    buffer = Image.fromarray(image).convert('RGB').tobytes()
    bitmap = wx.BitmapFromBuffer(width, height, buffer)
    return bitmap

class RealTimeSaliency(wx.Frame):
    # ----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Real-time saliency", size=(600, 400))

        self.SetMinClientSize((600, 400))
        self.on_update = None

        panel = wx.Panel(self, wx.ID_ANY)
        self.img_viewer = ImgViewPanel(self)


        self.index = 0
        self.list_ctrl = wx.ListCtrl(panel,
                                     style=wx.LC_REPORT)
        self.search_ctrl = wx.TextCtrl(panel, value='Search', size=(200, 25))
        self.search_ctrl.Bind(wx.EVT_TEXT, self.on_search)


        self.list_ctrl.InsertColumn(0, 'Class name', width=200)

        self.slider_ctrl = wx.Slider(panel, value=4, minValue=-2, maxValue=11, style=wx.SL_MIN_MAX_LABELS|wx.SL_VALUE_LABEL)
        self.slider_ctrl.Bind(wx.EVT_SCROLL, self.on_slide)
        self.info = wx.StaticText(panel)

        self.show_items_that_contain()

        btn = wx.Button(panel, label='Choose')
        btn.Bind(wx.EVT_BUTTON, self.choose_class)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(panel, 1,  wx.ALL | wx.EXPAND, 5)
        hsizer.Add(self.img_viewer, 2, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(hsizer)


        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.info, 0, wx.EXPAND, 5)
        sizer.Add(self.slider_ctrl, 0, wx.EXPAND, 5)
        sizer.Add(self.list_ctrl, 5, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.search_ctrl, 0)
        sizer.Add(btn, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(sizer)
        wx.CallLater(100, self.update)

    def on_slide(self, event):
        global CONFIDENCE
        CONFIDENCE = self.slider_ctrl.GetValue()

    def update(self):
        self.info.SetLabel('Showing: %s\nConfidence:' % CLASS_ID_TO_NAME[TO_SHOW])
        if self.on_update is not None:
            self.on_update()
        wx.CallLater(100, self.update)

    def on_search(self, event):
        self.show_items_that_contain(self.search_ctrl.GetValue())

    def show_items_that_contain(self, text=''):
        self.list_ctrl.DeleteAllItems()
        i = 0
        for name in CLASS_ID_TO_NAME.values():
            if text.lower().strip() in name.lower():
                self.list_ctrl.InsertItem(i, name)
                i += 1

    def choose_class(self, event):
        global TO_SHOW
        TO_SHOW = CLASS_NAME_TO_ID[self.list_ctrl.GetItem(self.list_ctrl.GetFocusedItem()).GetText()]


class ImgViewPanel(wx.Panel):
    def __init__(self, parent):
        self.parent = parent
        self.dialog_init_function = False
        self.dialog_out = False
        super(ImgViewPanel, self).__init__(parent, -1)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.img_now = None
        self.changed = False
        self.change_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        self.update()

    def bind_mouse(self, click, move, release, scroll):
        self.Bind(wx.EVT_LEFT_DOWN, click)
        self.Bind(wx.EVT_MOTION, move)
        self.Bind(wx.EVT_LEFT_UP, release)
        self.Bind(wx.EVT_MOUSEWHEEL, scroll)

    def update(self):
        if not self.changed:
            self.changed = True
            self.Refresh()
            self.Update()
        if self.dialog_init_function:
            try:
                # dialog_init_function must be a funtion that takes the parent as arg
                # and returns wx dialog object
                dialog = self.dialog_init_function(self)
                dialog.ShowModal()
                self.dialog_out = dialog.GetValue()
                self.dialog_out = self.dialog_out if self.dialog_out else True
                dialog.Destroy()

            except Exception, err:
                print 'Could not open the dialog!', err
            self.dialog_init_function = False
        wx.CallLater(15, self.update)

    def on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        dc.DrawBitmap(self.img_now, 0, 0)

    def change_frame(self, image):
        '''image must be PIL or wx image'''
        s = self.GetSize()
        x = s.x
        y = s.y
        image = cv2.resize(image, (x, y), interpolation=cv2.INTER_LINEAR)
        self.img_now = numpy_to_wx(image)

        self.changed = False


class RT:
    DELAY_SMOOTH = 0.85
    def __init__(self, processor, batch_size=1, view_size=(324, 324)):
        '''
        How it works?
        The images are continuously captured and added to the queue together with their frame timestamps.
        Another thread processes the images in the queue by passing them to the processor function.
        The processor function operates on batches of images: it takes a numpy array of shape (batch_size, H, W, 3) where H, W is the native resolution of the camera
        and must return a numpy array of shape (batch_size, ?, ?, 3) - note the size of the image does not matter as it will be resized to the view_size anyway.
        Images are normalized between -1 and 1!
        If the queue grows faster than the we can process the images then we will skip images and processor function will be given only
        every Nth image in the outstanding queue.

        Finally the processed images are placed on the display queue together with their timestamps and the display thread takes care of displaying images
        at the correct times by estimating the current overall delay and fps.
        For example if the processor takes 1 second to process one image then the dealy will be 1 second and the resulting fps will be 1.
        '''
        self.batch_size = batch_size
        self.processor = processor
        self.cam = None
        self.req_queue = Queue()
        self.display_queue = Queue()
        self.delay = 0.
        self.time_per_frame = 0.
        self.show_image = None


    def start(self):
        if self.cam is None:
            self.cam = cv2.VideoCapture(0)
        self._stop = False
        # start transformer and display services
        tr = threading.Thread(target=self.transform)
        dis = threading.Thread(target=self.display)
        tr.daemon = True
        dis.daemon = True
        tr.start()
        dis.start()
        self._get_next_frame()

    def _get_next_frame(self):
        ret_val, img = self.cam.read()
        img = cv2.flip(img, 1)
        self.req_queue.put((time.time(), img))

    def stop(self):
        self._stop = True
        time.sleep(1.)

    def transform(self):
        while not self._stop:
            if self.req_queue.qsize()< self.batch_size:
                time.sleep(POLL_DELAY)
                continue
            to_proc = []
            while not self.req_queue.empty():
                to_proc.append(self.req_queue.get(timeout=0.1))

            if len(to_proc) > self.batch_size:
                # usual case, take self.batch_size equally separated items
                sep = int(len(to_proc) / self.batch_size)
                old = len(to_proc)
                to_proc = to_proc[:sep*self.batch_size:sep]
                assert len(to_proc) == self.batch_size

            imgs = np.concatenate(tuple(np.expand_dims(e[1], 0) for e in to_proc), 0)
            done_imgs = ((self.processor(imgs/(255./2) - 1.) + 1) * (255./2.)).astype(np.uint8)

            for e in xrange(len(done_imgs)):
                im = done_imgs[e]
                t = to_proc[e][0]
                self.display_queue.put((t, im))

    def display(self):
        last_frame = time.time()
        while not self._stop:
            if self.display_queue.empty():
                time.sleep(POLL_DELAY)
                continue
            creation_time, im = self.display_queue.get(timeout=11)
            self.delay = self.DELAY_SMOOTH*self.delay + (1.-self.DELAY_SMOOTH)*(time.time() - creation_time)
            while time.time() < creation_time + self.delay:
                time.sleep(POLL_DELAY)
            self.time_per_frame = 0.9*self.time_per_frame + 0.1*(time.time() - last_frame)
            if self.show_image is not None:
                self.show_image(im)
            last_frame = time.time()

    @property
    def fps(self):
        return 1./self.time_per_frame



from saliency_eval import get_pretrained_saliency_fn
def get_proc_fn(cuda=False):
    fn = get_pretrained_saliency_fn(cuda=cuda)

    def proc(ims):
        sq = square_centrer_crop_resize_op(np.squeeze(ims, 0), (224, 224))
        sq = np.transpose(sq, (2, 0, 1))
        mask = fn(sq, TO_SHOW, CONFIDENCE)[0].cpu().data.numpy()
        sq = sq*(1-mask)
        return np.expand_dims(np.transpose(sq, (1, 2, 0)), 0)
    return proc

def square_centrer_crop_resize_op(im, size):
    short_edge = min(im.shape[:2])
    yy = int((im.shape[0] - short_edge) / 2)
    xx = int((im.shape[1] - short_edge) / 2)
    max_square = im[yy: yy + short_edge, xx: xx + short_edge]
    return cv2.resize(max_square, size, interpolation=cv2.INTER_LINEAR)



if __name__ == "__main__":
    app = wx.App(False)
    frame = RealTimeSaliency()
    a = RT(get_proc_fn(cuda=False))
    a.start()
    a.show_image = frame.img_viewer.change_frame
    frame.on_update = a._get_next_frame
    frame.Show()
    app.MainLoop()


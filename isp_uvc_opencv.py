import random
import sys
import time
import cv2
import numpy as np
import dmcam
import platform


def dei_isp_wait(dev, sleep_ms=20, timeout_ms=1000):
    ''' wait device ready when do write or read'''
    while True:
        v = int(dev.get(cv2.CAP_PROP_CONTRAST))
        if (v & 1) == 1:  # still busy
            if timeout_ms <= 0:
                return False
            time.sleep(sleep_ms / 1000.0)
            timeout_ms -= sleep_ms
        else:
            break

    return True


def dei_isp_write(dev, addr, val):
    ''' operation to write isp register '''
    addr |= 1  # set valid bit
    dev.set(cv2.CAP_PROP_GAMMA, (addr >> 16) & 0xffff)
    dev.set(cv2.CAP_PROP_SATURATION, val & 0xffff)
    dev.set(cv2.CAP_PROP_SHARPNESS, (val >> 16) & 0xffff)

    dev.set(cv2.CAP_PROP_CONTRAST, addr & 0xffff)  # trig op

    if not dei_isp_wait(dev):
        raise Exception("wait isp op timeout!")


def dei_isp_read(dev, addr):
    ''' operation to read isp register '''
    addr |= 3  # set valid bit and rd bit
    dev.set(cv2.CAP_PROP_GAMMA, (addr >> 16) & 0xffff)
    dev.set(cv2.CAP_PROP_CONTRAST, addr & 0xffff)  # trig op
    if not dei_isp_wait(dev):
        raise Exception("wait isp op timeout!")
    v0 = int(dev.get(cv2.CAP_PROP_SATURATION))
    v1 = int(dev.get(cv2.CAP_PROP_SHARPNESS))
    return v0 | (v1 << 16)


def isp_dev_wait_api_ready(dev, sleep_ms=10, timeout_ms=1000):
    ''' wait isp device ready to accept simple API commands'''
    while True:
        if dei_isp_read(dev, 0x0020) & (1 << 8):
            if timeout_ms <= 0:
                return False
            time.sleep(sleep_ms / 1000.0)
            timeout_ms -= sleep_ms
        else:
            break

    return True


def isp_dev_cmd_char(dev, ch) -> bool:
    ''' write single char command via simple API'''
    isp_dev_wait_api_ready(dev)
    dei_isp_write(dev, 0xa800, (ch & 0xff) | 0xae00)
    dei_isp_write(dev, 0x0020, 1)  # trigger op
    return True


def isp_dev_set_fps(dev, fps: int) -> bool:
    isp_dev_wait_api_ready(dev)
    dei_isp_write(dev, 0xa820, fps | (1 << 31))
    dei_isp_write(dev, 0x0020, 1)  # trigger op
    return True


def isp_dev_set_expo(dev, us: int) -> bool:
    isp_dev_wait_api_ready(dev)
    dei_isp_write(dev, 0xa81c, us | (1 << 31))
    dei_isp_write(dev, 0x0020, 1)  # trigger op
    return True


def isp_dev_set_stream_on(dev, on: bool) -> bool:
    isp_dev_wait_api_ready(dev)
    dei_isp_write(dev, 0xa818, (1 if on else 0) | (1 << 31))
    dei_isp_write(dev, 0x0020, 1)  # trigger op
    return True


def switch_depth_ir_frame(dev):
    # dev.set(cv2.CAP_PROP_BUFFERSIZE, 8)  # if fps is to high, this should enlage
    dev.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    dev.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    dev.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', '1', '6', ' '))  # needed for linux
    dev.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # disable windows convert to RGB
    print("Width = ", dev.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height = ", dev.get(cv2.CAP_PROP_FRAME_HEIGHT))


tof_cam_idx = 0

try:
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()
    # print(graph.get_input_devices())  # list of camera device
    tof_cam_idx = graph.get_input_devices().index("DepEye USB Video")
    print("-> find DepEye USB cam: index=%d" % tof_cam_idx)
except ImportError:
    print("-> Use default cam idx=%d" % tof_cam_idx)
except ValueError:
    print("-> No DepEye USB cam detected")
    sys.exit(255)

if platform.system().lower() == 'windows':
    dev = cv2.VideoCapture(tof_cam_idx, cv2.CAP_DSHOW)  #
else:
    dev = cv2.VideoCapture(tof_cam_idx)  #

switch_depth_ir_frame(dev)

isp_dev_set_fps(dev, 30)

print(cv2.__version__)
print(" ---- Key help ----")
print("   q : exit")
print("   ` : Dual-freq enable")
print("   1 : First freq mode enable")
print("   2 : Second freq mode enable")
print("   0 : Toggle normal/shuffule ")
print("   m : Toggle medianfilter  ")
print("   M : Toggle flynoise filter  ")  # under linux : waitkey not support uppercase
print("   a : Toggle auto exposure  ")
print("   f/F : -/+ fps value ")  # under linux : waitkey not support uppercase
print("   -/+ : -/+ expo value ")

ts_pinfo = time.time()

while (1):
    # get raw frame
    ret, raw_frame = dev.read()

    if not ret:
        time.sleep(0.1)
        continue
    # print(ret, raw_frame.shape)
    # convert to depth and ir frame
    dep_ir_frame = np.frombuffer(raw_frame, dtype=np.uint16).reshape(480, 1280)
    depth, ir = dep_ir_frame[:, 0::2], dep_ir_frame[:, 1::2]
    # cv2.imshow("DepEye ToF UVC Demo", ir.astype(np.uint8))

    # convert depth and ir to pseudo rgb
    cnt, dep_bgr_img = dmcam.cmap_dist_u16_to_RGB(len(depth.flatten()) * 3, depth.flatten(), dmcam.DMCAM_CMAP_OUTFMT_BGR, 0, 4000, None)
    cm_cfg = dmcam.cmap_cfg()
    cm_cfg.color_palette, cm_cfg.histeq_en = dmcam.DMCAM_CMAP_GRAY, True
    cnt, ir_bgr_img = dmcam.cmap_dist_u16_to_RGB(len(ir.flatten()) * 3, ir.flatten(), dmcam.DMCAM_CMAP_OUTFMT_BGR, 0, 256, cm_cfg)

    # concat ir depth horizontally
    bgr_img = np.hstack([dep_bgr_img.reshape((480, -1)), ir_bgr_img.reshape((480, -1))])
    bgr_img = bgr_img.reshape(480, 1280, 3)

    # show a frame
    cv2.imshow("DepEye ToF UVC Demo", bgr_img)

    # --- handle key ---
    k = cv2.waitKey(2) & 0xff
    if k == ord('q'):
        break
    elif k in [ord(x) for x in ['`', '1', '2', 't', '0', 'm', 'M', 'a', 'f', 'F', '+', '-', 'g', '3']]:
        isp_dev_cmd_char(dev, k)
    elif k == ord(' '):
        expo_us = random.randint(100, 1500)
        print(' * random setting expo to %d' % expo_us)
        isp_dev_set_expo(dev, expo_us)

    # --- print status of device ---
    if time.time() - ts_pinfo > 3.0 or k != 255:
        ts_pinfo = time.time()
        # get DEV_STAT
        dev_info = dei_isp_read(dev, 0xa82c)
        shuffle_en, stream_on, fps_raw, fps_dep = (dev_info >> 28) & 3, (dev_info >> 24) & 1, (dev_info >> 12) & 0xfff, dev_info & 0xfff

        # get expo info
        expo_info = dei_isp_read(dev, 0xa830)
        ae_en, expo_us = (expo_info >> 28) & 1, expo_info & 0x0fffffff

        # get temp info
        temp_info = dei_isp_read(dev, 0xa834)
        t_ldd, t_sensor = (temp_info >> 16) & 0xffff, temp_info & 0xffff
        # get freq-info
        freq_info = dei_isp_read(dev, 0xa838)
        f0_mhz, f1_mhz = freq_info & 0xff, (freq_info >> 16) & 0xff

        print(f" * expo={expo_us}[a={ae_en}], freq={f0_mhz}/{f1_mhz}Mhz [shuffle={shuffle_en}], tcb/tib={t_sensor / 10.0}/{t_ldd / 10.0}, fps-raw/dep={fps_raw}/{fps_dep}")

dev.release()
cv2.destroyAllWindows()

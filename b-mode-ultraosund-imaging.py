import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import hilbert
from scipy.fft import fft, ifft



FanImage_org = 'Fan.png'
FanImage_linear = 'FanImage_linear.png'

def get_b_mode():

    data = scipy.io.loadmat('homwork_3/rf2017.mat')
    
    img = data['rf_signal']
    img = np.transpose(img)
    aline = 300
    dataLength = 5000
    samplingRate = 1000000000  # 1GHz
    interval = 0.00005  # 50 um
    vpp = 200

    samplePoint = 4096

    for index in range(60,66):
        t = np.arange(0,dataLength,1)
        fig, axs = plt.subplots()
        axs.set_title("Image of Time domain A-line "+str(index))
        axs.plot(t,img[index], color='C0')
        axs.set_xlabel("Time(μs)")
        axs.set_ylabel("Amplitute(mV)")
        plt.axis('on')
        plt.savefig('Image of Time domain A-line '+str(index)+' frequency ', bbox_inches='tight', pad_inches=0.0)
        plt.plot(t,img[index])
        plt.show()
    distance = interval * aline * 1000                    # mm
    depth = dataLength / samplingRate / 2 * 1540 * 1000   # mm

    spectrum = abs(hilbert(img))

    cal =spectrum/(np.min(spectrum))
    compression = 20*np.log10(cal)

    compressed_signal = compression-(np.max(compression))
    us_image=(compressed_signal+42)/42*255
    

    img = np.transpose(us_image)
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(img, cmap=plt.gray(), vmin=0, vmax=255, aspect='auto', extent=[0, distance, depth, 0])
    plt.axis('off')
    plt.savefig('B-mode', bbox_inches='tight', pad_inches=0.0)
   
   
    ax.set_aspect('equal')
    ax.set_xlabel('Distance of Imaging (mm)')
    ax.set_ylabel('Depth of Imaging (mm)')
    ax.set_title('B-mode')
    fig.colorbar(im, orientation='vertical')
    plt.axis('on')
    plt.savefig('B-mode(legend)', bbox_inches='tight', pad_inches=0.0)


def change_square_to_fan(inputFileName:str, outputFileName:str, angle:int): 
    im = Image.open(inputFileName)
 
    mode = im.mode
    w, h = im.size
    rows, cols = int(np.ceil(h)), int(np.ceil(2 * h * np.sin(np.radians(angle))))
    cols += cols % 2
    padding = 2
    
    im_squ = np.array(im)
    # im_fan = np.zeros((rows,cols,im_squ.shape[2]),dtype=np.uint8)
    im_fan = np.zeros((rows + padding, cols + padding, im_squ.shape[2]), dtype=np.uint8)
    
    alpha = np.radians(np.linspace(-angle, angle, w))
    for i in range(w):
        d = np.cos(alpha[i]) * rows
        lats = np.int_(np.linspace(0, d, h)).astype(np.int)
        d = np.sin(alpha[i]) * rows
        lons = np.int_(np.linspace(cols / 2, cols / 2 + d, h)).astype(np.int)
        im_fan[(lats, lons)] = im_squ[:, i]
    
    im = Image.fromarray(im_fan, mode=mode)
    im.save('BeforeInterpolate_rf_' + outputFileName)

    for row in range(rows):
        ps, pe = 0, 0
        for col in range(cols):
            if im_fan[row, col, 3] > 0:
                if ps == 0:
                    ps, pe = col, col
                else:
                    pe = col
        for col in range(ps - 1, pe):
            if im_fan[row, col, 3] == 0:
                im_fan[row, col] = im_fan[row, col - 1]

    im = Image.fromarray(im_fan, mode=mode)
    im.save('AfterInterpolate_rf_' + outputFileName)

def linear_inter(fn_squ:str, fn_fan, angle:int, 
                 r0=0, k=2, top=True, 
                 rotate=0):
    im_pil = Image.open(fn_squ)
    im = np.array(im_pil)
    h,w,d = im.shape
    print('k',k)
    if r0>0:
        bg = np.ones((r0,w,d),dtype=np.uint8)
        im = np.append(bg,im,axis = 0) if top else np.append(im,bg,axis=0)

    h,w,d = im.shape
    r = 2*h-1
    im_fan = np.zeros((r,r,d),dtype=np.uint8)
    idx = np.arange(h) if top else np.arange(h)[::-1]
    alpha =  np.radians(np.linspace(-angle/2,angle/2,k*w))
    for i in range(k*w):
        rows = ((np.ceil(np.cos(alpha[i])*idx)) +r/2).astype('int')
        cols = (np.int32(np.ceil(np.sin(alpha[i])*idx)) +r/2).astype('int')
        im_fan[(rows,cols)] = im[:,i//k]
    if 360 > angle and angle > 180:
        im_fan = im_fan[int(h*(1-np.sin(np.radians((angle/2-90))))):]

    if not top:
        im_fan = im_fan[::-1]
    im_out= Image.fromarray(im_fan,mode=im_pil.mode)
    im_out.save(fn_fan)
    img = Image.open(fn_fan)
    dst = img.rotate(rotate)
    dst.save(fn_fan)     

if __name__ == '__main__':
    get_b_mode()
    change_square_to_fan('B-mode.png',FanImage, 60)
    linear_inter('B-mode.png', FanImage_linear, 120, 1, 200, False)
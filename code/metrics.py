import numpy as np
#import skimage.measure as measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio  as compare_psnr
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu
from glob import glob
from tifffile import imread
import tifffile
from natsort import natsorted
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os

class Metrics(object):
    def __init__(self, phantoms, architecture, pos, it, depth, width, dilation, base_path, run_folder):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """
        self.phantoms= phantoms
        self.pos = pos
        self.it =it
        self.depth= depth  # cause we dont iteratate
        self.architecture= architecture
        self.mode = 'horizontal'
        self.classes = 3  # Classes used for Otsu thresholding
        self.ROI = [slice(40, 130), slice(60, 441), slice(60, 441)]
        self.dilation=dilation
        self.width=width
        self.run_folder= run_folder
        self.base_path=base_path
        
        self.path_to_radial_imgs = base_path + "ConeBeamResults/" + self.run_folder  # they are also named with walnut not phantom #'/bigstore/jordi/Walnuts/results/predictions/Radial'
        self.path_to_rad2ax_imgs = self. base_path + "ReconResults/" +self.run_folder  # /bigstore/jordi/Walnuts/results/predictions/Radial2Axial/
        print(self.path_to_rad2ax_imgs)
        self.path_to_hor_imgs = '/bigstore/jordi/Walnuts/results/predictions/Horizontal/'
        
        self.results_path = base_path + "MetricsResults/"  + self.run_folder  # + '*'  #.format(self.architecture, self.pos, self.width, self.depth,self.dilation, self.it))
        self.compute_ssim = True
        self.compute_mse = True
        self.compute_dsc = True
        self.compute_psnr = True
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    def otsu(self, image, num_classes=None):
        '''
        image: str or array
        '''
        if num_classes is None:
            num_classes = self.classes
        # Read image if it is a string
        if type(image) == str:
            image = imread(image)

        thresholds = threshold_multiotsu(image, classes=num_classes)
        segm = np.digitize(image, bins=thresholds)


        return segm

    def ssim(self, image, ref, roi=None):
        '''
        image: str or array
        ref: str or array
        '''

        # Read image if it is a string
        
        if type(image) == str:
            image = imread(image).astype(float)

        if type(ref) == str:
            ref = imread(ref).astype(float)

        if roi != None:
            image = image[roi]
            ref = ref[roi]

        max_ref = ref.max()
        min_ref = ref.min()

        ranges = max_ref - min_ref
        ssim = compare_ssim(image, ref, data_range=ranges) #measure.compare_ssim() but deprecated

        return ssim

    def mse(self, image, ref, roi=None):
        '''
        image: str or array
        ref: str or array
        '''

        # Read image if it is a string
        if type(image) == str:
            image = imread(image)

        if type(ref) == str:
            ref = imread(ref)

        if roi != None:
            image = image[roi]
            ref = ref[roi]

        mse = compare_mse(image, ref)
        return mse

    def dsc(self, image, ref, roi=None):
        '''
        image: str or array
        ref: str or array
        '''
        # Read image if it is a string
        if type(image) == str:
            image = imread(image)

        if type(ref) == str:
            ref = imread(ref)

        if roi != None:
            image = image[roi]
            ref = ref[roi]

        dsc = np.zeros([self.classes-1])

        for i in np.unique(image):
            if i == 0:
                continue

            im1 = np.asarray(image) == i
            im2 = np.asarray(ref) == i

            if im1.shape != im2.shape:
                raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

            # Compute Dice coefficient
            intersection = np.logical_and(im1, im2)
            dsc[i-1] =  2. * intersection.sum() / (im1.sum() + im2.sum())
        return dsc

    def psnr(self, image, ref, roi=None):
        '''
        image: str or array
        ref: str or array
        '''

        # Read image if it is a string
        if type(image) == str:
            image = imread(image).astype(float)

        if type(ref) == str:
            ref = imread(ref).astype(float)

        if roi != None:
            image = image[roi]
            ref = ref[roi]

        psnr = compare_psnr(image, ref)

        return psnr

    def plot(self, ssim=None, mse=None, dsc_low=None, dsc_high= None, psnr=None, x1=0, x2=0, title=''):

        lst = [ssim, mse, dsc_low, dsc_high, psnr]
        total = sum(y is not None for y in lst)
        counter = 0

        plt.figure()

        if ssim is not None:
            counter += 1
            x = np.linspace(0, len(ssim[0]), num=len(ssim[0]))
            ax1  = plt.subplot(total,1,counter)
            FDK, = plt.plot(x, ssim[0], label= 'FDK')
            IR, = plt.plot(x, ssim[1], label= 'IR')
            RAD, = plt.plot(x, ssim[3], label= 'CNN - Radial')
            HOR, = plt.plot(x, ssim[2], label= 'CNN - Horizontal')
            ax1.fill_between(x, y1=0, y2=1,where=(x>x1-25) & (x<x1+25), facecolor='grey', alpha = 0.3)
            ax1.axvline(x1, color='black', linewidth=1, linestyle='--')
            ax1.axvline(x2, color='black', linewidth=1, linestyle='--')
            plt.ylabel('SSIM')
            plt.ylim(0.00,1.00)
            plt.legend([FDK, IR, HOR, RAD], ['FDK', 'IR', ' CNN - horizontal', 'CNN - radial'], bbox_to_anchor=(0, 1.02, 1, .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)


        if mse is not None:
            counter += 1
            ax2 = plt.subplot(total,1,counter)
            x = np.linspace(0, len(mse[0]), num=len(mse[0]))
            FDK, = plt.plot(x, mse[0], label= 'FDK')
            IR, = plt.plot(x, mse[1], label= 'IR')
            RAD, = plt.plot(x, mse[3], label= 'CNN - Radial')
            HOR, = plt.plot(x, mse[2], label= 'CNN - Horizontal')
            ax2.fill_between(x, y1=0, y2=1, where=(x>x1-25) & (x<x1+25), facecolor='grey', alpha=0.3)
            ax2.axvline(x1, 0, 1, color='black', linewidth=1, linestyle='--')
            ax2.axvline(x2, 0, 1, color='black', linewidth=1, linestyle='--')
            print(x1,x2)
            plt.ylabel('MSE')
            plt.ticklabel_format(style= 'sci', axis='y',scilimits=(0,0))
            plt.yscale('log')

        if dsc_low is not None:
            counter += 1
            plt.subplot(total,1,counter)
            x = np.linspace(0, len(dsc_low[0]), num=len(dsc_low[0]))
            FDK, = plt.plot(x, dsc_low[0], label='FDK')
            IR, = plt.plot(x, dsc_low[1], label='IR')
            RAD, = plt.plot(x, dsc_low[3], label='CNN - Radial')
            HOR, = plt.plot(x, dsc_low[2], label='CNN - Horizontal')
            plt.ylabel('DSC-LD')
            plt.ylim(0.00, 1.0)

        if dsc_high is not None:
            counter += 1
            plt.subplot(total,1,counter)
            x = np.linspace(0, len(dsc_high[0]), num=len(dsc_high[0]))
            FDK, = plt.plot(x, dsc_high[0], label='FDK')
            IR, = plt.plot(x, dsc_high[1], label='IR')
            RAD, = plt.plot(x, dsc_high[3], label='CNN - Radial')
            HOR, = plt.plot(x, dsc_high[2], label='CNN - Horizontal')
            plt.ylabel('DSC-HD')
            plt.ylim(0.00, 1.0)

        if psnr is not None:
            counter += 1
            plt.subplot(total,1,counter)
            x = np.linspace(0, len(psnr[0]), num=len(psnr[0]))
            FDK, = plt.plot(x, psnr[0], label= 'FDK')
            IR, = plt.plot(x, psnr[1], label= 'IR')
            RAD, = plt.plot(x, psnr[3], label= 'CNN - Radial')
            HOR, = plt.plot(x, psnr[2], label= 'CNN - Horizontal')
            plt.ylabel('PSNR')


        plt.xlabel('Slice number')
        plt.savefig(self.results_path  + title + '.pdf')#can add figures folder here still
        plt.cla()
        print('plotted')

    '''' PLOT PER SLICE - AVERAGE'''

    def plot_per_phantom(self, phantom, ssim_per_slice, mse_per_slice, dsc_per_slice, psnr_per_slice, x1, x2):
        ph_index=0  #theres only one  phantom
        ssim_plot = [np.mean(ssim_per_slice, axis=0)[0, :], np.mean(ssim_per_slice, axis=0)[1, :],
                     np.mean(ssim_per_slice, axis=0)[2, :], np.mean(ssim_per_slice, axis=0)[3, :]]
        mse_plot = [np.mean(mse_per_slice, axis=0)[0, :], np.mean(mse_per_slice, axis=0)[1, :],
                    np.mean(mse_per_slice, axis=0)[2, :], np.mean(mse_per_slice, axis=0)[3, :]]
        dsc_low_plot = [dsc_per_slice[ph_index, 0], dsc_per_slice[ph_index, 2], dsc_per_slice[ph_index, 4],
                        dsc_per_slice[ph_index, 6]]
        dsc_high_plot = [dsc_per_slice[ph_index, 1], dsc_per_slice[ph_index, 3], dsc_per_slice[ph_index, 5],
                         dsc_per_slice[ph_index, 7]]

        psnr_plot = [psnr_per_slice[ph_index, 0], psnr_per_slice[ph_index, 1], psnr_per_slice[ph_index, 2],
                     psnr_per_slice[ph_index, 3]]
        print( phantom, str(phantom))
        phantom = str(phantom)
        np.savetxt(self.results_path + 'phantom_'+ phantom + '_dsc_low_plot.txt', dsc_low_plot)
        np.savetxt(self.results_path + 'phantom_'+ phantom + '_dsc_high_plot.txt', dsc_high_plot)
        np.savetxt(self.results_path + 'phantom_'+ phantom + '_ssim_plot.txt', ssim_plot)
        np.savetxt(self.results_path + 'phantom_'+ phantom + '_psnr_plot.txt', psnr_plot)
        np.savetxt(self.results_path + 'phantom_'+ phantom + '_mse_plot.txt', mse_plot)
        self.plot(ssim=ssim_plot, mse=mse_plot, dsc_low=dsc_low_plot, dsc_high=dsc_high_plot, psnr=psnr_plot, x1=x1,
                  x2=x2, title='Evaluation_metrics_phantom{}_{}'.format(phantom, self.architecture))
        print('Per Phantom Plots done and results saved.')

    ''' SAVE IMAGES '''

   #save specific slices
    def save_slice(self,phantom, slice, gold_standard_imgs, rad2ax_imgs):
        slice=slice
        phantom=str(phantom)
        tifffile.imsave(self.results_path + str(slice) + '_vol_gs_otsu.tiff', self.otsu(gs[slice, :, :]).astype(np.float32))
        tifffile.imsave(self.results_path + 'phantom' +phantom + '_' + str(slice) + '_gs_otsu.tiff', self.otsu(gold_standard_imgs[slice]).astype(np.float32))
        tifffile.imsave(self.results_path + 'phantom' + phantom + '_' +  str(slice) + '_rad2ax_otsu.tif', self.otsu(rad2ax_imgs[slice]).astype(np.float32))
        #tifffile.imsave(self.results_path + str(slice) + '_horizontal_otsu.tiff', self.otsu(hor_imgs[slice]).astype(np.float32))


    '''MAIN FILE'''
    def calculate_metrics(self):

        SSIM = np.zeros([len(self.phantoms), 4])
        DSC = np.zeros([len(self.phantoms), 4 * (self.classes-1)])
        MSE = np.zeros([len(self.phantoms), 4])
        PSNR =  np.zeros([len(self.phantoms), 4])

        SSIM_ROI = np.zeros([len(self.phantoms), 4])
        DSC_ROI = np.zeros([len(self.phantoms), 4 * (self.classes-1)])
        MSE_ROI = np.zeros([len(self.phantoms), 4])
        PSNR_ROI = np.zeros([len(self.phantoms), 4])

        ssim_per_slice = np.zeros([len(self.it), 4, 501])
        mse_per_slice = np.zeros([len(self.it), 4, 501])
        dsc_per_slice = np.zeros([len(self.it), 4 * (self.classes-1), 501])
        psnr_per_slice = np.zeros([len(self.it), 4, 501])


        for z, phantom in enumerate(self.phantoms):
            for it_idx, ii in enumerate(self.it):
                print(phantom, 'Phantom')

                gold_standard_imgs = natsorted(glob('/bigstore/felix/Walnuts/Walnut{}/Reconstructions/nnls_pos123*'.format(phantom)))
                input_imgs = natsorted(glob('/bigstore/felix/Walnuts/Walnut{}/Reconstructions/fdk_pos{}*'.format(phantom,self.pos)))
                ir_imgs = natsorted(glob('/bigstore/jordi/Walnuts/IR_reconstruction/Walnut{}/Reconstructions/nnls_pos1*'.format(phantom)))
                rad2ax_imgs = natsorted(glob(self.path_to_rad2ax_imgs +'{}_pos{}_it{}_depth{}_phantom{}/*'.format(self.architecture, self.pos, ii, self.depth, phantom)))

                it_hor=2    #needs to be adjusted based on reconstructions
                hor_imgs = natsorted(
                    glob(self.path_to_hor_imgs + '{}_pos{}_it{}_*_phantom{}/*'.format(self.architecture, self.pos, it_hor, phantom)))

                assert len(gold_standard_imgs) == len(input_imgs) == len(rad2ax_imgs) == len(hor_imgs) == len(ir_imgs)

                inp_vol = np.zeros([len(input_imgs), 501, 501])
                gs = np.zeros([len(input_imgs), 501, 501])
                ir_vol = np.zeros([len(input_imgs), 501, 501])
                rad2ax_vol = np.zeros([len(input_imgs), 501, 501])
                horizontal_vol = np.zeros([len(input_imgs), 501, 501])

                imgs = [input_imgs,ir_imgs,  hor_imgs, rad2ax_imgs]
                volumes = [inp_vol, ir_vol, horizontal_vol, rad2ax_vol]


                for j, img in enumerate(imgs):
                    print(j, 'j', len(img))
                    # Calculate metrics per slice
                    for jj, gs_img in enumerate(gold_standard_imgs):
                        image = img[jj]

                        if self.compute_ssim == True:
                            ssim_per_slice[it_idx, j, jj] = self.ssim(img[jj], gs_img)
                        if self.compute_mse == True:
                            mse_per_slice[it_idx, j, jj] = self.mse(img[jj], gs_img)
                        if self.compute_dsc == True:
                            dsc_per_slice[it_idx, j * (self.classes-1) : (j+1) * (self.classes-1), jj] = self.dsc(self.otsu(img[jj]), self.otsu(gs_img))  # we get 2 outputs for 3 clasess

                        if self.compute_psnr == True:
                            psnr_per_slice[it_idx, j, jj] = self.psnr(img[jj], gs_img)

                        # Create volumes
                        inp_vol[jj, :, :] = imread(input_imgs[jj])
                        gs[jj, :, :] = imread(gold_standard_imgs[jj])
                        ir_vol[jj, :, :] = imread(ir_imgs[jj])
                        rad2ax_vol[jj, :, :] = imread(rad2ax_imgs[jj])
                        horizontal_vol[jj, :, :] = imread(hor_imgs[jj])

                # Calculate metrics for volumes as a whole
                for k, vol in enumerate(volumes):
                    if self.compute_ssim == True:
                        SSIM[z, k] = self.ssim(vol, gs)
                        SSIM_ROI[z, k] = self.ssim(vol, gs, roi=self.ROI)
                    if self.compute_mse == True:
                        MSE[z, k] = self.mse(vol, gs)
                        MSE_ROI[z, k] = self.mse(vol, gs, roi = self.ROI)
                    if self.compute_dsc == True:
                        DSC[z, k * (self.classes-1): (k+1)*(self.classes-1)] = self.dsc(self.otsu(vol, num_classes=self.classes), self.otsu(gs, num_classes=self.classes))
                        print(DSC.shape, 'shape DSC')
                        DSC_ROI[z, k * (self.classes-1): (k+1)*(self.classes-1)] = self.dsc(self.otsu(vol, num_classes=self.classes), self.otsu(gs, num_classes=self.classes), roi = self.ROI)
                    if self.compute_psnr == True:
                        PSNR[z, k] = self.psnr(vol, gs)
                        PSNR_ROI[z,k] = self.psnr(vol, gs, roi=self.ROI)

                print('Processed iteration {}'.format(ii))

            #save selected slices, optional
            slices=[100]
            if len(slices)>0:
                for slice in slices:
                    self.save_slice(phantom, slice, gold_standard_imgs, rad2ax_imgs)


            make_per_phantom_plots= True
            if make_per_phantom_plots is True:
                otsu_gs = self.otsu(gs, num_classes=self.classes)
                otsu_rad2ax = self.otsu(rad2ax_vol, num_classes=self.classes)
                otsu_hor = self.otsu(horizontal_vol, num_classes=self.classes)

                vol = otsu_gs + otsu_rad2ax + otsu_hor
                vol[vol != 6] = 0

                for x in range(vol.shape[0]):
                    if np.sum(vol[x, :, :]) > 1000:
                        x1 = x
                        break
                x2 = max(np.nonzero(vol)[0])

                self.plot_per_phantom( phantom, ssim_per_slice, mse_per_slice, dsc_per_slice, psnr_per_slice, x1, x2)


        comparisons = ['input vs gold_standard', 'iterative vs gold standard', 'radial2axial vs gold standard', 'horizontal vs gold standard']


        for b in range(len(self.phantoms)):
            for a in range(4):
                print("SSIM Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), SSIM[b,a])
                print("DSC Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), DSC[b,a])
                print("MSE Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), MSE[b,a])
                print("PSNR Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), PSNR[b,a])


                print("SSIM-ROI Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), SSIM_ROI[b,a])
                print("DSC-ROI Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), DSC_ROI[b,a])
                print("MSE-ROI Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), MSE_ROI[b,a])
                print("PSNR-ROI Phantom {} - IT{} - {}:".format(self.phantoms[b], ii, comparisons[a]), PSNR_ROI[b,a])


        comparisons_header= "input  iterative   horizontal  radial2axial"
        #save txt files
        if self.compute_ssim == True:
            np.savetxt(self.results_path +'SSIM_{}_it{}.txt'.format(self.architecture, ii), SSIM, header = comparisons_header)
            np.savetxt(self.results_path +'SSIM_ROI_{}_it{}.txt'.format(self.architecture, ii), SSIM_ROI, header = comparisons_header)

        if self.compute_mse == True:
            np.savetxt(self.results_path + 'MSE_{}_it{}.txt'.format(self.architecture, ii), MSE, header = comparisons_header)
            np.savetxt(self.results_path + 'MSE_ROI_{}_it{}.txt'.format(self.architecture, ii), MSE_ROI, header = comparisons_header)

        if self.compute_dsc == True:
            np.savetxt(self.results_path + 'DSC_{}_it{}.txt'.format(self.architecture, ii), DSC, header = comparisons_header)
            np.savetxt(self.results_path + 'DSC_ROI_{}_it{}.txt'.format(self.architecture, ii), DSC_ROI, header =comparisons_header)

            np.savetxt(self.results_path + 'DSC_low_{}_it{}.txt'.format(self.architecture, ii), DSC[:,::2], header = comparisons_header)
            np.savetxt(self.results_path + 'DSC_ROI_low_{}_it{}.txt'.format(self.architecture, ii), DSC_ROI[:,::2], header = comparisons_header)
            np.savetxt(self.results_path + 'DSC_high_{}_it{}.txt'.format(self.architecture, ii), DSC[:,1::2], header =comparisons_header)
            np.savetxt(self.results_path + 'DSC_ROI_high_{}_it{}.txt'.format(self.architecture, ii), DSC_ROI[:,1::2], header =comparisons_header)


        if self.compute_psnr == True:
            np.savetxt(self.results_path + 'PSNR_{}_it{}.txt'.format(self.architecture, ii), PSNR, header = comparisons_header)
            np.savetxt(self.results_path + 'PSNR_ROI_{}_it{}.txt'.format(self.architecture, ii), PSNR_ROI, header = comparisons_header)
        DSC_low=DSC[:,6]
        DSC_ROI_low=DSC_ROI[:,6]
        DSC_high=DSC[:,7]
        DSC_ROI_high=DSC_ROI[:,7]
        return(SSIM[:,3][0],SSIM_ROI[:,3][0],MSE[:,3][0], MSE_ROI[:,3][0], DSC_low[0], DSC_ROI_low[0], DSC_high[0], DSC_ROI_high[0], PSNR[:,3][0], PSNR_ROI[:,3][0])

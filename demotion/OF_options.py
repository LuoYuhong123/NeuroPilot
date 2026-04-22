import os
import json
import numpy as np
import tifffile
from typing import List, Union, Callable
from tifffile import TiffFile
from demotion import get_displacements
from demotion import imregister_wrapper
from scipy.ndimage import gaussian_filter
from demotion.imresize import imresize



class VideoFileReader:
    def __init__(self):
        self.frame_count = None
        self.bitdepth = None
        self.mat_data_type = None

        self.downsampling_kernel = None
        self.m = None
        self.n = None

        self.buffer_size = 500
        self.bin_size = 1

    @property
    def datatype(self):
        raise NotImplementedError

    # @property
    # def current_frame(self):
    #     raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def read_batch(self):
        raise NotImplementedError

    def read_frames(self, idx):
        raise NotImplementedError

    def has_batch(self):
        raise NotImplementedError

    def batches_left(self):
        raise NotImplementedError

    # @current_frame.setter
    # def current_frame(self, x):
    #     self._current_frame = x

    @staticmethod
    def get_mat_data_type(bitdepth, signed=False):
        if bitdepth == 8:
            return 'int8' if signed else 'uint8'
        elif bitdepth in {10, 12, 16}:
            return 'int16' if signed else 'uint16'
        else:
            return 'float64'

    def set_downsampling_kernel(self):
        self.downsampling_kernel = np.ones((1, 1, self.bin_size)) / self.bin_size
        self.reset()

    def bin_buffer(self, buffer):
        if self.bin_size > 1:
            if buffer.shape[2] <= self.bin_size:
                buffer = np.mean(buffer, axis=2)
            else:
                buffer = np.convolve(buffer, self.downsampling_kernel, mode='same')
                buffer = buffer[:, :, int(np.ceil(self.bin_size / 2))::self.bin_size]
        return buffer

    def get_width(self):
        return self.n

    def get_height(self):
        return self.m


class TIFSTACKFileReader(VideoFileReader):
    datatype = 'TIFSTACK'

    def __init__(self, input_file, buffer_size=None, bin_size=None, deinterleave=1):
        super().__init__()
        self.deinterleave = deinterleave
        #self.images_raw = tifffile.imread(input_file)
        all_frames = []
        with tifffile.TiffFile(input_file) as tif:
            for s in tif.series:
                data = tif.asarray(series=s)
                all_frames.append(data)
        self.images_raw= np.concatenate(all_frames,axis=0)
        print('Successfully read tiff file: ',input_file)
        self.m = self.images_raw.shape[1]
        self.n = self.images_raw.shape[2]
        self.t = self.images_raw.shape[0]
        self.images = self.images_raw
        self.length = self.m
        if self.m != self.n:
            self.length = max(self.m,self.n)
            self.images = np.zeros((self.t,self.length,self.length))
            for i in range(self.t):
                self.images[i] = imresize(self.images_raw[i], output_shape=(self.length, self.length))
            print(f'Resize images to shape:({self.length},{self.length})')
        self.data_type = self.images.dtype

        #self.tif = TiffFile(input_file)
        #self.bitdepth = self.tif.pages[0].tags['BitsPerSample'].value
        #self.mat_data_type = self.get_mat_data_type(self.bitdepth) #signed=(self.tif.pages[0].tags['SampleFormat'].value == 3))  # 3 means Int type

        #img_info = self.tif.pages[0].imagej_metadata
        self.frame_count = self.t#len(self.tif.pages)



        self.buffer_size = buffer_size or self.buffer_size
        self.bin_size = bin_size or self.bin_size

        self.current_frame = 0

    def read_batch(self):
        if self.current_frame >= self.frame_count:
            return None

        n_elem_left = min(self.buffer_size * self.bin_size, self.frame_count - self.current_frame)
        st = self.current_frame
        ed = self.current_frame + n_elem_left
        buffer = self.images[st:ed,:,:]
        self.current_frame = self.current_frame + n_elem_left
        # buffer = np.zeros((self.length, self.length, n_elem_left),dtype=self.data_type)
        #
        # for i in range(n_elem_left):
        #     self.current_frame += 1
        #     current_idx = self.current_frame
        #     page = self.images[current_idx - 1,:,:]
        #     buffer[:, :, i] = np.array(page).astype(self.data_type)#.asarray()#.astype(self.mat_data_type)

        return buffer.transpose((1,2,0)),st,ed

    def read_frames(self, idx):
        assert np.all(np.array(idx) <= self.frame_count)
        st = min(idx)
        ed = max(idx) + 1
        buffer = self.images[st:ed,:,:]
        # n_elements = len(idx)
        # buffer = np.zeros((self.length, self.length, n_elements),dtype=self.data_type)
        #
        # for i, frame_idx in enumerate(idx):
        #     current_idx = frame_idx
        #     page = self.images[current_idx - 1,:,:]
        #     buffer[:, :, i] = np.array(page).astype(self.data_type)#.asarray()#.astype(self.mat_data_type)

        return buffer.transpose((1,2,0)),self.m,self.n

    def has_batch(self):
        # print('current:',self.current_frame)
        # print('total:',self.frame_count)
        return self.current_frame < self.frame_count

    def batches_left(self):
        return int(np.ceil((self.frame_count - self.current_frame) / (self.buffer_size * self.bin_size)))

    def close(self):
        self.tif.close()







class OFOptions:
    def __init__(self, **kwargs):
        # Set default values
        self.input_file = '2p_1.tiff'
        self.supported_extensions = ['.MDF', '.tif', '.tiff', '.mat']
        self.ext_map = ['MDF', 'TIFSTACK', 'TIFSTACK', 'MAT']
        self.quality_setting_old = 'quality'
        self.output_path = 'results'
        self.output_format = 'MAT'
        self.channel_idx = None
        self.output_file_name = None
        self.output_file_writer = None
        self.alpha = 1.5
        self.weight = [[0.5], [0.5]]
        self.levels = 100
        self.min_level = -1
        self.quality_setting = 'quality'
        self.eta = 0.8
        self.update_lag = 5
        self.iterations = 50
        self.a_smooth = 1
        self.a_data = 0.45
        self.sigma = [1, 1, 0.1]
        self.bin_size = 1
        self.buffer_size = 400
        self.verbose = False
        self.reference_frames = list(range(50, 500))
        self.save_meta_info = False
        self.save_w = False
        self.output_typename = 'double'
        self.channel_normalization = 'joint'
        self.interpolation_method = 'cubic'
        self.update_reference = True
        self.preproc_funct = None
        self.p = kwargs
        self.raw_images_dir = None

        # Initialize the parameters based on keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Ignoring invalid option '{key}'")

        # Handle `quality_setting` initialization
        if self.min_level != -1:
            self.quality_setting = 'custom'

        if 'quality_setting' in kwargs:
            self.min_level = self.get_min_level()
            self.quality_setting_old = kwargs.get('quality_setting')

    def set_output_path(self, output_path: str):
        if isinstance(output_path, str):
            self.output_path = output_path
        else:
            raise ValueError("output_path must be a string")

    def set_input_file(self, input_file: str):
        self.input_file = input_file

    def set_reference_frames(self, reference_frames: Union[List[int], str]):
        self.check_reference(reference_frames)
        self.reference_frames = reference_frames

    def set_alpha(self, alpha: Union[float, List[float]]):
        self.check_alpha(alpha)
        self.alpha = alpha

    def set_sigma(self, sigma: List[float]):
        self.check_sigma(sigma)
        self.sigma = sigma

    def set_weight(self, weight: List[float]):
        self.check_weight(weight)
        self.weight = weight / np.sum(weight) if len(weight) > 1 else weight

    def set_output_format(self, output_format: str):
        valid_formats = ['TIFF', 'HDF5', 'MAT', 'MULTIFILE_TIFF', 'MULTIFILE_MAT', 'MULTIFILE_HDF5', 'CAIMAN_HDF5',
                         'BEGONIA']
        if output_format not in valid_formats:
            raise ValueError(f"{output_format} is not a valid output format")
        self.output_format = output_format

    def set_quality_setting(self, quality_setting: str):
        valid_settings = ['quality', 'balanced', 'fast', 'custom']
        if quality_setting not in valid_settings:
            raise ValueError(f"{quality_setting} is not a valid quality setting")
        if quality_setting != 'custom':
            self.quality_setting_old = quality_setting
        self.quality_setting = quality_setting

    def set_min_level(self, min_level: float):
        if min_level >= 0:
            self.quality_setting = 'custom'
        else:
            self.quality_setting = self.quality_setting_old
        self.min_level = min_level

    #@property
    def get_min_level(self):
        if self.min_level == -1:
            if self.quality_setting == 'quality':
                print('quality_setting: quality')
                return 0
            elif self.quality_setting == 'balanced':
                print('quality_setting: balanced')
                return 4
            elif self.quality_setting == 'fast':
                print('quality_setting: fast')
                return 6
            elif self.quality_setting == 'custom':
                print('quality_setting: custom')
                return max(self.min_level, 0)
        return self.min_level

    def check_reference(self, reference_frames):
        if isinstance(reference_frames, (str, list)):
            return
        raise ValueError("Invalid reference frames format")

    def check_alpha(self, alpha):
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("Alpha must be a positive number")

    def check_sigma(self, sigma):
        if not isinstance(sigma, list) or len(sigma) != 3 or any(s <= 0 for s in sigma):
            raise ValueError("Sigma must be a list of 3 positive values")

    def check_weight(self, weight):
        if not isinstance(weight, list) or not all(isinstance(w, (int, float)) for w in weight):
            raise ValueError("Weight must be a list of numbers")

    def get_video_file_reader(self):
        # Placeholder for file reader (using TIFSTACK as an example)
        # Implement a video reader depending on your use case
        return TIFSTACKFileReader(self.input_file, self.buffer_size, self.bin_size)



    def get_reference_frame(self, video_file_reader):
        if isinstance(self.reference_frames, (list, np.ndarray)):
            tmp,m,n = video_file_reader.read_frames(self.reference_frames)  # Convert to numpy array

            data_type = tmp.dtype

            #print(data_type)
            # cv2.imshow('tmp',tmp[:,:,0] / np.max(tmp[:,:,0]))
            # cv2.waitKey(-1)
            if tmp.shape[2] == 1:  # If it's a single channel/frame
                reference = tmp
                return reference

            weight_2d = self.weight

            if not self.verbose:
                print("Pre-registering reference frames...")

            # Applying Gaussian filter (replacing imgaussfilt3 in MATLAB)
            # c1 = np.matlib.normalize(
            #     ndimage.gaussian_filter(tmp, sigma=self.sigma))  # Normalize first (equivalent to mat2gray)
            sigma = self.sigma + np.array([1, 1, 0.5])  # Assuming obj.sigma is a 1D array
            # sigma = tuple(sigma[0])  
            print("tmp.shape =", np.asarray(tmp).shape, "sigma =", sigma)
            filtered_tmp = gaussian_filter(tmp, sigma=sigma)
            c1 = (filtered_tmp - np.min(filtered_tmp)) / (np.max(filtered_tmp) - np.min(filtered_tmp))

            c_reg_tmp, w = self.compensate_sequence(c1, np.mean(c1, axis=2), tmp, np.mean(tmp, axis=2))

            # Set the reference as the mean of the compensated frames
            reference = np.mean(c_reg_tmp, axis=2)
            w_init = np.mean(w,axis=3)
            #self.reference_frames = reference

            if not self.verbose:
                print("Finished pre-registration of the reference frames...")
            return reference,w_init,data_type,m,n
        
        
    def compensate_sequence(self,c, c_ref, c_raw, c_ref_raw):
        # print(self.alpha)
        vargin = {'weight':self.weight, 'alpha':self.alpha+2,
                  'levels':self.levels, 'min_level':self.min_level,
                  'eta':self.eta, 'update_lag':self.update_lag,
                  'iterations':self.iterations,'a_smooth':self.a_smooth,
                  'a_data':self.a_data}
        w = get_displacements.get_displacements(c,c_ref,vargin)
        c_comp = np.zeros_like(c,dtype=c.dtype)

        for i in range(c.shape[2]):
            c_comp[:,:,i] = imregister_wrapper.imregister_wrapper_w(c_raw[:,:,i],w[:,:,:,i],c_ref_raw)

        #tifffile.imwrite('test_init_warped.tiff',c_comp.transpose((2,0,1)))
        return c_comp,w

    def load_options(self, settings_file: str):
        if not os.path.isfile(settings_file):
            raise FileNotFoundError(f"{settings_file} not found")

        with open(settings_file, 'r') as file:
            options = json.load(file)

        for key, value in options.items():
            setattr(self, key, value)

    def save_options(self, settings_path: str):
        options = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        with open(settings_path, 'w') as file:
            json.dump(options, file, indent=4)




if __name__ == '__main__':
    import cv2
    import time
    input_path = '2p_1.tiff'
    output_path = 'test.tiff'
    options = OFOptions(input_file = input_path,
                        output_file_name = output_path,
                        output_format = 'TIFF',
                        alpha = 1.5,
                        sigma = [2,2,0.1],
                        quality_setting='fast',
                        bin_size=1,
                        buffer_size=200,
                        reference_frames=list(range(400, 600))
                        )
    video_file_reader = options.get_video_file_reader()
    c_ref_raw = options.get_reference_frame(video_file_reader)
    print(c_ref_raw.shape)
    # vargin ={'weight':[[0.5],[0.5]],'alpha':3.5,'levels':100.0,'min_level':4.0,'eta':0.8,'update_lag':5.0,'iterations':50.0,'a_smooth':1.0,'a_data':0.45}
    # st_time = time.time()
    # w = get_displacements.get_displacements(tmp,tmp_ref,vargin)
    # ed_time = time.time()
    # print(w.shape)
    # print('time cost:',ed_time - st_time)

    #c_ref_raw = options.get_reference_frame(video_file_reader)

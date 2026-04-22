import os
import numpy as np
import scipy.ndimage as ndi
import tifffile
from scipy.io import savemat
import cv2
from demotion.get_displacements import get_displacements,get_displacements_2
from demotion.imregister_wrapper import imregister_wrapper, imregister_wrapper_w
from demotion.imresize import imresize
import demotion.flow_viz



def compensate_recording(options):
    # Check if output directory exists, if not create it
    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    if not options.verbose:
        print(
            f"\nStarting compensation with quality setting {options.quality_setting} and min_level = {options.min_level}.")
        print(
            f"Set the quality setting to fast, balanced or quality or increase min_level for fast, approximated solutions.")
        print(f"Output format is {options.output_format}\n")
    all_frames = []
    with tifffile.TiffFile(options.raw_images_dir) as tif:
        for s in tif.series:
            data = tif.asarray(series=s)
            all_frames.append(data)
    raw_images_ = np.concatenate(all_frames,axis=0)
    raw_images_ = raw_images_.transpose((1,2,0))
    #raw_images_ = tifffile.imread(options.raw_images_dir).transpose((1,2,0))
    video_file_reader = options.get_video_file_reader()
    #print(raw_images_.shape)
    # Get reference frame if not passed as argument

    c_ref_raw,w_init,data_type,m,n = options.get_reference_frame(video_file_reader)

    if m != n:
        length = max(m,n)
        raw_images = np.zeros((length,length,raw_images_.shape[2]))
        for i in range(raw_images_.shape[2]):
            raw_images[:,:,i] = imresize(raw_images_[:,:,i],output_shape=(length,length),method='bicubic')
        
    else:
        length = m
        raw_images = raw_images_
################################# 上采样  ############################################
    
    # length = 400
    # raw_images = np.zeros((length,length,raw_images_.shape[2]))
    # for i in range(raw_images_.shape[2]):
    #         raw_images[:,:,i] = imresize(raw_images_[:,:,i],output_shape=(length,length),method='bicubic')

    # Setting the channel weight
    weight = options.weight

    i = 0
    while video_file_reader.has_batch():
        i += 1
        buffer,st,ed = video_file_reader.read_batch()
        # print(buffer.shape)
       

        raw_buffer = raw_images[:,:,st:ed]
        if i == 1:
            tmp = ndi.gaussian_filter(buffer, options.sigma)
            min_ref = np.min(tmp)
            max_ref = np.max(tmp)

            c_ref = ndi.gaussian_filter(c_ref_raw, options.sigma[:2])
            c_ref = (c_ref - min_ref) / (max_ref - min_ref)


            if not options.verbose:
                print('Done pre-registration to get w_init.')
        
###############################  上采样  #####################################################
        # buffer_upsampled = np.zeros((length,length,raw_buffer.shape[2]))
        # c_ref_upsampled = np.zeros((length,length))
        # c_ref_upsampled = imresize(c_ref,output_shape=(length,length),method='bicubic')
        # for j in range(raw_buffer.shape[2]):
        #     buffer_upsampled[:,:,j] = imresize(buffer[:,:,j],output_shape=(length,length),method='bicubic')
        # c_ref_upd, w_init, w, raw_reg = get_eval(options, buffer_upsampled, c_ref_upsampled, c_ref_raw, w_init, weight,raw_buffer,length)
        c_ref_upd, w_init, w, raw_reg,pred_w = get_eval(options, buffer, c_ref, c_ref_raw, w_init, weight,raw_buffer,length)
        if options.update_reference:
            c_ref = c_ref_upd



        # Write compensated frames to the video file writer
        if i ==1:
            warped_images = np.array(raw_reg)
            warp_flows = np.array(w)
            pred_flows = np.array(pred_w)
        else:
            warped_images = np.concatenate((warped_images,raw_reg),axis=2)
            warp_flows = np.concatenate((warp_flows,w),axis=3)
            pred_flows = np.concatenate((pred_flows,pred_w),axis=3)
        #video_file_writer.write_frames(c_reg)

        if not options.verbose:
            print(f"Finished batch {i}, {video_file_reader.batches_left()} batches left.")
    warped_images_output = warped_images
    
    # tifffile.imwrite(fr'E:\flow_deepcad\plot_code\Fig2_validation\warp_naomi_flows.tif',warp_flows)
    # tifffile.imwrite(fr'E:\flow_deepcad\plot_code\Fig2_validation\pred_naomi_flows.tif',pred_flows)



    if m!=n:
        warped_images_output = np.zeros((m,n,warped_images.shape[2]))
        
        for i in range(warped_images.shape[2]):
            warped_images_output[:,:,i] = imresize(warped_images[:,:,i],output_shape=(m,n))
            
        print('Finished resizing to original shape!')

####################### 下采样 ####################################
    # warped_images_output = np.zeros((m,n,warped_images.shape[2]))
    
    # for i in range(warped_images.shape[2]):
    #     warped_images_output[:,:,i] = imresize(warped_images[:,:,i],output_shape=(m,n),method='bicubic')
        
    # print('Finished resizing to original shape!')

    
    if data_type == 'uint16':
        print('demotion save as uint16')
        warped_images_output = np.clip(warped_images_output, 0, 65535)
        warped_images_output = warped_images_output.astype('uint16')

    elif data_type == 'int16':
        print('demotion save as int16')
        warped_images_output = np.clip(warped_images_output, -32767, 32767)
        warped_images_output = warped_images_output.astype('int16')

    elif data_type == 'uint8':
        print('demotion save as uint8')
        warped_images_output = np.clip(warped_images_output, 0, 255)
        warped_images_output = warped_images_output.astype('uint8')

    else:
        print('demotion default save as uint16')
        warped_images_output = warped_images_output.astype('uint16')
    tifffile.imwrite(options.output_file_name,warped_images_output.transpose((2,0,1)))
    #return pred_flows
    # Close video file writer
    #video_file_writer.close()




def get_eval(options, buffer, c_ref, c_ref_raw, w_init, weight,raw_buffer,length):
    c1 = np.array(buffer).astype(np.float64)
    c1 = (c1 - c1.min()) / (c1.max() - c1.min())  # Normalize using mat2gray
    c1 = ndi.gaussian_filter(c1, options.sigma)
    c1 = (c1 - c1.min()) / (c1.max() - c1.min())

    if c_ref is None:
        c_ref = np.mean(c1, axis=2)

    if w_init is None:
        w_init = np.zeros((c_ref.shape[0], c_ref.shape[1], 2))
    vargin = {'weight': weight, 'alpha': options.alpha, 'levels': options.levels,
              'min_level': options.min_level, 'eta': options.eta,
              'update_lag': options.update_lag, 'iterations': options.iterations, 'a_smooth': options.a_smooth,
              'a_data': options.a_data,'uv':(w_init[:, :, 0], w_init[:, :, 1])}
    w = get_displacements(c1, c_ref, vargin)
    ################################################################################
    w_pred = w # get_displacements_2(c1,c_ref,vargin)
    ###################################################################################
    n_ref_frames = min(20000, c1.shape[2] - 1)
    c_ref = np.mean(compensate_sequence_uv(c1[:, :, -n_ref_frames:], np.mean(c1, axis=2), w[:, :, :, -n_ref_frames:]),axis=2)

    if options.preproc_funct:
        buffer = options.preproc_funct(buffer)

    if options.output_typename:
        buffer = buffer.astype(options.output_typename)
    
    
    #c_reg = compensate_sequence_uv(buffer, c_ref_raw, w)
    raw_reg = compensate_sequence_uv(raw_buffer,np.mean(raw_buffer,axis=2),w)
    # cv2.imshow('warped',c_reg[:,:,0] / np.max(c_reg[:,:,0]))
    # cv2.waitKey(-1)
    w_end = np.mean(w[:, :, :, -20:], axis=3) if raw_reg.shape[2] > 1000 else np.mean(w, axis=3)

    #return mean_disp, max_disp, mean_div, mean_translation, c_reg, c_ref, w_end, w
    return c_ref,w_end,w,raw_reg,w_pred


def compensate_sequence_uv(c,c_ref,w):
    c_comp = np.zeros_like(c, dtype=c.dtype)
    for i in range(c.shape[2]):
        c_comp[:, :, i] = imregister_wrapper_w(c[:, :, i], w[:, :, :, i], c_ref)
    return c_comp


def get_mean_divergence(w):
    divergence = np.zeros(w.shape[3])

    u = w[:, :, 0, :]
    v = w[:, :, 1, :]

    for i in range(w.shape[3]):
        w_x, _ = np.gradient(u[:, :, i])
        _, w_y = np.gradient(v[:, :, i])

        divergence[i] = np.mean(w_x + w_y)

    return divergence


def get_mean_translation(w):
    u = np.mean(np.mean(w[:, :, 0, :], axis=1), axis=2)
    v = np.mean(np.mean(w[:, :, 1, :], axis=1), axis=2)

    return np.sqrt(u ** 2 + v ** 2)

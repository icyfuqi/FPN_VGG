function [output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file)
% [output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%首先初始化RPN的测试网络；然后产生不同长宽的全零图像并进行前向传播；记录每个输入图像大小对应的conv5_3大小；重置caffe。

%    caffe.init_log(fullfile(pwd, 'caffe_log'));
    %初始化RPN的测试网络
    caffe_net = caffe.Net(test_net_def_file, 'test');
    
     % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end
    
	%产生不同长宽的全零图像并进行前向传播
    input = 100:conf.max_size;
    output_w = nan(size(input));
    output_h = nan(size(input));
    for i = 1:length(input)
        s = input(i);
        im_blob = single(zeros(s, s, 3, 1));
        net_inputs = {im_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        caffe_net.forward(net_inputs);
        
		%记录每个输入图像大小对应的conv5_3大小
        cls_score = caffe_net.blobs('proposal_cls_score').get_data();
        output_w(i) = size(cls_score, 1);
        output_h(i) = size(cls_score, 2);
    end
    
    output_width_map = containers.Map(input, output_w);
    output_height_map = containers.Map(input, output_h);
    
	%重置caffe
    caffe.reset_all(); 
end
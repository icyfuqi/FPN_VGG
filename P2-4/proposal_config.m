function conf = proposal_config(varargin)
% conf = proposal_config(varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ip = inputParser;
    
    %% training
    ip.addParamValue('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);
                                    
    % whether drop the anchors that has edges outside of the image boundary
	%在训练阶段是否去掉超出图像边界的anchors 
    ip.addParamValue('drop_boxes_runoff_image', ...
                                        true,           @islogical);
    
    % Image scales -- the short edge of input image
    %短边缩放后最小值	
    ip.addParamValue('scales',          600,            @ismatrix);
    % Max pixel size of a scaled input image
	%长边缩放后最大值
    ip.addParamValue('max_size',        1000,           @isscalar);
    % Images per batch, only supports ims_per_batch = 1 currently
	%训练时每个batch中的图像个数，当前只支持每次输入一幅图像 
    ip.addParamValue('ims_per_batch',   1,              @isscalar);
    % Minibatch size
	%训练时每个batch中的正负样本个数 
    ip.addParamValue('batch_size',      256,            @isscalar);
    % Fraction of minibatch that is foreground labeled (class > 0)
	%batch_size中正样本的比例，如果正样本个数不足，则添加负样本 
    ip.addParamValue('fg_fraction',     0.5,           @isscalar);
    % weight of background samples, when weight of foreground samples is
    % 1.0
	%计算损失时每个负样本的权值，正样本权值全为1 
    ip.addParamValue('bg_weight',       1.0,            @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    %与ground-truth的iou大于阈值0.7的roi作为正样本
	ip.addParamValue('fg_thresh',       0.7,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
	%与ground-truth的iou在阈值0-0.3之间的roi作为负样本
    ip.addParamValue('bg_thresh_hi',    0.3,            @isscalar);
    ip.addParamValue('bg_thresh_lo',    0,              @isscalar);
    % mean image, in RGB order
	%图像均值
    ip.addParamValue('image_means',     128,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',     true,           @islogical);
    % Stride in input image pixels at ROI pooling level (network specific)
    % 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
	%feature map到原图映射的变化率为16，由池化降维算出，与其步长选择有关
	%VGG中conv5_3相比于输入图像缩小了16倍，也就是相邻两个点之间的stride=16 
    ip.addParamValue('feat_stride',     4,             @isscalar);
    % train proposal target only to labled ground-truths or also include
    % other proposal results (selective search, etc.)
    ip.addParamValue('target_only_gt',  true,           @islogical);

    % random seed                    
    ip.addParamValue('rng_seed',        6,              @isscalar);

    
    %% testing
    ip.addParamValue('test_scales',     600,            @isscalar);
    ip.addParamValue('test_max_size',   1000,           @isscalar);
    ip.addParamValue('test_nms',        0.3,            @isscalar);
    ip.addParamValue('test_binary',     false,          @islogical);
    ip.addParamValue('test_min_box_size',16,            @isscalar);
    ip.addParamValue('test_drop_boxes_runoff_image', ...
                                        false,          @islogical);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end
function dataset = voc2007_trainval(dataset, usage, use_flip)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit                      = voc2007_devkit();

switch usage
    case {'train'}
	    %imdb_from_voc()函数:从VOC数据集中加载训练数据，如果是第一次运行该函数，结果会被保存到cache中，以后再运行这个程序时，不用重新计算，直接在cache里加载上次结果
        dataset.imdb_train    = {  imdb_from_voc(devkit, 'trainval', '2007', use_flip) };
		%加载roi数据，提供了四种获取proposal的方法：with_selective_search、with_edge_box 、with_self_proposal以及RPN方法。如果第一次运行，则会计算初始的roi，并且将得到的roi保存到cache中，以后再运行程序时，直接从roi的cache中读取roi数据
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test     = imdb_from_voc(devkit, 'trainval', '2007', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end
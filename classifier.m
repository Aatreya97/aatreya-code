function [bestScore, category] = classifier(image_folder, image_name)
    % setup MatConvNet. Your path might be different.
    run ../matconvnet-master/matconvnet-master/matlab/vl_setupnn ;

    % load the 221MB pre-trained CNN
    net = load('imagenet-vgg-f.mat') ;

    % load and preprocess an image
    im = imread(fullfile(image_folder, image_name)) ;
    im_ = single(im) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    category = net.meta.classes.description{best};
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',category, best, bestScore)) ;
    if ~exist(fullfile(image_folder, 'result'), 'dir')
        mkdir(fullfile(image_folder, 'result'));
    end
    saveas(gcf, fullfile(image_folder, ['result\' image_name]));
end

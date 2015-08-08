#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  bool is_triplet = this->layer_param_.image_data_param().is_triplet();
  const int lines_size = lines_.size();
  if (this->phase_ == TRAIN&&is_triplet) 
  {
    labels_filenames.clear();
    std::map<int, vector<std::string> >::iterator iter;      
    for(int item_id = 0; item_id < lines_size; ++item_id)
    {
    //LOG(INFO) << "item_id" << item_id << " images.";
      int label_id = lines_[item_id].second;
      iter = labels_filenames.find(label_id);
      if(iter == labels_filenames.end())
      {
        vector<std::string> filenames(1,lines_[item_id].first);
        labels_filenames.insert(std::pair<int,vector<std::string> >(label_id,filenames));
      }
      else
      {
        iter->second.push_back(lines_[item_id].first);
      }
    }

    LOG(INFO) << "item_id = " << labels_filenames.size()<< " images.";

    for(int item_id = 0; item_id < idx.size(); ++item_id)
    {
      idx[item_id].clear();    
    }
    idx.clear();
    idx.resize(labels_filenames.size());

    for(int item_id = 0; item_id < lines_size; ++item_id)
    {
      int label_id = lines_[item_id].second;
      idx[label_id].push_back(item_id); 
    }

  

    int tmp_count = 0 ;
    for(int item_id = 0; item_id < idx.size(); ++item_id)
    {
      tmp_count = tmp_count + idx[item_id].size();    
    }
  }

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  bool is_triplet = image_data_param.is_triplet() ;

  int num_anchors = image_data_param.num_anchors() ;
  int num_pos_imgs = image_data_param.num_pos_imgs() ;
  vector<int>tmp_idx(batch_size);
  tmp_idx.clear();
  vector<int>anchors_idx(num_anchors);
  anchors_idx.clear();

  if (this->phase_ == TRAIN&&is_triplet)
  {
    int nValidAnchors = 0;
    while(nValidAnchors < num_anchors) 
    {
      int people_id = rand() % (idx.size());

      if(idx[people_id].size() < num_pos_imgs)
      {
        continue;
        //people_id = rand() % (idx.size());     
        //LOG(ERROR) << people_id<<" the num_pos_imgs is too big";   
      }
      
      if( anchors_idx.end() != find(anchors_idx.begin(), anchors_idx.end(), people_id))
      {
        continue;
      }
      ++nValidAnchors;
      
      anchors_idx.push_back(people_id);

      std::random_shuffle(idx[people_id].begin(), idx[people_id].end());

      for (int j = 0; j < num_pos_imgs; ++j)
      {
        int t_idx = idx[people_id][j] ;
        tmp_idx.push_back(t_idx);
      }
    }


    int item_num = tmp_idx.size();

    while(item_num < batch_size)
    {
      int item_id_tmp = rand() % (lines_size);    
      bool is_same = false;
      for (int j = 0; j < anchors_idx.size(); ++j)
      {
        if(lines_[item_id_tmp].second == anchors_idx[j])
        {
          is_same = true;
        }
      } 

      if (!is_same)
      {
        tmp_idx.push_back(item_id_tmp);
        item_num++;
      }
    }
  }





  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe

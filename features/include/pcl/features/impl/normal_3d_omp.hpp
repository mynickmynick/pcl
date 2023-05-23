/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_FEATURES_IMPL_NORMAL_3D_OMP_H_
#define PCL_FEATURES_IMPL_NORMAL_3D_OMP_H_


#include <pcl/features/normal_3d_omp.h>
#include <omp.h>
#include <thread>

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT> void
pcl::NormalEstimationOMP<PointInT, PointOutT>::setNumberOfThreads (unsigned int nr_threads)
{
  if (nr_threads == 0)
#ifdef _OPENMP
    threads_ = omp_get_num_procs();
#else
    threads_ = 1;
#endif
  else
    threads_ = nr_threads;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT> void
pcl::NormalEstimationOMP<PointInT, PointOutT>::computeFeatureMP (PointCloudOut &output)
{
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  pcl::Indices nn_indices (k_);
  std::vector<float> nn_dists (k_);

  output.is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
  if (input_->is_dense)
  {
#pragma omp parallel for \
  default(none) \
  shared(output) \
  firstprivate(nn_indices, nn_dists) \
  num_threads(threads_)
    // Iterating over the entire index vector
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t> (indices_->size ()); ++idx)
    {
      Eigen::Vector4f n;
      if (this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0 ||
          !pcl::computePointNormal (*surface_, nn_indices, n, output[idx].curvature))
      {
        output[idx].normal[0] = output[idx].normal[1] = output[idx].normal[2] = output[idx].curvature = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      output[idx].normal_x = n[0];
      output[idx].normal_y = n[1];
      output[idx].normal_z = n[2];

      flipNormalTowardsViewpoint ((*input_)[(*indices_)[idx]], vpx_, vpy_, vpz_,
                                  output[idx].normal[0], output[idx].normal[1], output[idx].normal[2]);

    }
  }
  else
  {
#pragma omp parallel for \
  default(none) \
  shared(output) \
  firstprivate(nn_indices, nn_dists) \
  num_threads(threads_)
    // Iterating over the entire index vector
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t> (indices_->size ()); ++idx)
    {
      Eigen::Vector4f n;
      if (!isFinite ((*input_)[(*indices_)[idx]]) ||
          this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0 ||
          !pcl::computePointNormal (*surface_, nn_indices, n, output[idx].curvature))
      {
        output[idx].normal[0] = output[idx].normal[1] = output[idx].normal[2] = output[idx].curvature = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      output[idx].normal_x = n[0];
      output[idx].normal_y = n[1];
      output[idx].normal_z = n[2];

      flipNormalTowardsViewpoint ((*input_)[(*indices_)[idx]], vpx_, vpy_, vpz_,
                                  output[idx].normal[0], output[idx].normal[1], output[idx].normal[2]);

    }
  }
}





template <typename PointInT, typename PointOutT> void
pcl::NormalEstimationOMP<PointInT, PointOutT>::computeFeatureThread (PointCloudOut &outp,
  const size_t i0, const size_t i1, const size_t t)
{
  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  pcl::Indices nn_indices (k_);
  std::vector<float> nn_dists (k_);

  shared_ptr<Indices> indices = make_shared<Indices>();
  indices->clear();
  indices->resize(this->indices_->size());
  std::copy(this->indices_->begin(), this->indices_->end(), indices->begin());
  
 // std::shared_ptr<pcl::search::KdTree<PointInT>> searcher_;
  // Initialize the search class

    //if (input_->isOrganized())
    //  searcher_.reset(new pcl::search::OrganizedNeighbor<PointInT>());
    //else
     // searcher_.reset(new pcl::search::KdTree<PointInT>());


  
  //PointCloudConstPtr input_=this->input_;
  const std::shared_ptr<const pcl::PointCloud<PointInT>> input = input_;//(new PointCloud(*input_)) ;
  const std::shared_ptr<const pcl::PointCloud<PointInT>> surface = surface_;//(new PointCloud(*surface_));

       PointCloudOut output;// (outp);
       output.resize(indices->size());
       output.width = outp.width;
       output.height = outp.height;
       searcher[t].setInputCloud(input, indices);

  // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
  if (input->is_dense)
  {

    // Iterating over the entire index vector
    for (std::ptrdiff_t idx = i0; idx < static_cast<std::ptrdiff_t> (i1); ++idx)
    {
      Eigen::Vector4f n;
      if (//this->searchForNeighbors ((*indices)[idx], search_parameter_, nn_indices, nn_dists) == 0 ||
        searcher[t].nearestKSearch((*input)[(*indices)[idx]], search_parameter_, nn_indices, nn_dists) == 0 ||
        !pcl::computePointNormal (*surface, nn_indices, n, output[idx].curvature))
      {
        output[idx].normal[0] = output[idx].normal[1] = output[idx].normal[2] = output[idx].curvature = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      output[idx].normal_x = n[0];
      output[idx].normal_y = n[1];
      output[idx].normal_z = n[2];

      flipNormalTowardsViewpoint ((*input)[(*indices)[idx]], vpx_, vpy_, vpz_,
        output[idx].normal[0], output[idx].normal[1], output[idx].normal[2]);

    }
  }
  else
  {
    // Iterating over the entire index vector
    for (std::ptrdiff_t idx = i0; idx < static_cast<std::ptrdiff_t> (i1); ++idx)
    {
      Eigen::Vector4f n;
      if (!isFinite ((*input)[(*indices)[idx]]) ||
        //this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0 ||
        searcher[t].nearestKSearch((*input)[(*indices)[idx]], search_parameter_, nn_indices, nn_dists) == 0 ||
        !pcl::computePointNormal (*surface, nn_indices, n, output[idx].curvature))
      {
        output[idx].normal[0] = output[idx].normal[1] = output[idx].normal[2] = output[idx].curvature = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      output[idx].normal_x = n[0];
      output[idx].normal_y = n[1];
      output[idx].normal_z = n[2];

      flipNormalTowardsViewpoint ((*input)[(*indices)[idx]], vpx_, vpy_, vpz_,
        output[idx].normal[0], output[idx].normal[1], output[idx].normal[2]);

    }
  }


  for (std::ptrdiff_t idx = i0; idx < static_cast<std::ptrdiff_t> (i1); ++idx)
    outp[idx] = output[idx];
  
}

template <typename PointInT, typename PointOutT> void
pcl::NormalEstimationOMP<PointInT, PointOutT>::computeFeature (PointCloudOut &output)
{

  output.is_dense = true;
  
  size_t chunk = indices_->size() / threads_;
  std::vector<std::thread>
#if __cplusplus> 201402L 
    alignas(std::hardware_destructive_interference_size) 
#endif 
    ThPool;
  size_t i0 = 0;
  size_t i1 = chunk;

  for (size_t t = 0; t < threads_; ++t)
  {
    if (t == threads_ - 1)
      i1 = indices_->size();

    ThPool.push_back( std::move( std::thread(&pcl::NormalEstimationOMP<PointInT, PointOutT>::computeFeatureThread,this,
      std::ref(output),
      i0, i1, t
    )));
    i0 += chunk;
    i1 += chunk;
  }

  for (size_t t = 0; t < threads_; ++t)
    ThPool[t].join();

}



template <typename PointInT, typename PointOutT> void
pcl::NormalEstimationOMP<PointInT, PointOutT>::computeMT(const
  PointCloudConstPtr& cloud, PointCloudOut& output)
{

  pcl::NormalEstimation<PointInT, PointOutT>::setInputCloud(cloud);
  /*if (!pcl::Feature<PointInT, PointOutT>::initCompute())
  {
    output.width = output.height = 0;
    output.clear();
    return;
  }*/



  {
    if (!pcl::PCLBase<PointInT>::initCompute())
    {
      PCL_ERROR("[pcl::%s::initCompute] Init failed.\n", getClassName().c_str());
      return ;
    }

    // If the dataset is empty, just return
    if (input_->points.empty())
    {
      PCL_ERROR("[pcl::%s::compute] input_ is empty!\n", getClassName().c_str());
      // Cleanup
      pcl::Feature<PointInT, PointOutT>::deinitCompute();
      return ;
    }

    // If no search surface has been defined, use the input dataset as the search surface itself
    if (!pcl::Feature<PointInT, PointOutT>::surface_)
    {
      pcl::Feature<PointInT, PointOutT>::fake_surface_ = true;
      pcl::Feature<PointInT, PointOutT>::surface_ = input_;
    }




    // Do a fast check to see if the search parameters are well defined
    if (Feature<PointInT, PointOutT>::search_radius_ != 0.0)
    {
      if (k_ != 0)
      {
        PCL_ERROR("[pcl::%s::compute] ", getClassName().c_str());
        PCL_ERROR("Both radius (%f) and K (%d) defined! ", Feature<PointInT, PointOutT>::search_radius_, k_);
        PCL_ERROR("Set one of them to zero first and then re-run compute ().\n");
        // Cleanup
        pcl::Feature<PointInT, PointOutT>::deinitCompute();
        return;
      }
      else // Use the radiusSearch () function
      {
        search_parameter_ = Feature<PointInT, PointOutT>::search_radius_;


      }
    }
    else
    {
      if (k_ != 0) // Use the nearestKSearch () function
      {
        search_parameter_ = k_;

      }
      else
      {
        PCL_ERROR("[pcl::%s::compute] Neither radius nor K defined! ", getClassName().c_str());
        PCL_ERROR("Set one of them to a positive number first and then re-run compute ().\n");
        // Cleanup
        pcl::Feature<PointInT, PointOutT>::deinitCompute();
        return;
      }
    }

  }



  // Copy the header
  output.header = input_->header;

  // Resize the output dataset
  if (output.size() != indices_->size())
    output.resize(indices_->size());

  // Check if the output will be computed for all points or only a subset
  // If the input width or height are not set, set output width as size
  if (indices_->size() != input_->points.size() || input_->width * input_->height == 0)
  {
    output.width = indices_->size();
    output.height = 1;
  }
  else
  {
    output.width = input_->width;
    output.height = input_->height;
  }
  output.is_dense = input_->is_dense;



  output.is_dense = true;

  size_t chunk = indices_->size() / threads_;
  std::vector<std::thread>
#if __cplusplus> 201402L 
    alignas(std::hardware_destructive_interference_size)
#endif 
    ThPool;
  size_t i0 = 0;
  size_t i1 = chunk;

  for (size_t t = 0; t < threads_; ++t)
  {
    if (t == threads_ - 1)
      i1 = indices_->size();

    ThPool.push_back(std::move(std::thread(&pcl::NormalEstimationOMP<PointInT, PointOutT>::computeFeatureThread, this,
      std::ref(output),
      i0, i1, t
    )));
    i0 += chunk;
    i1 += chunk;
  }

  for (size_t t = 0; t < threads_; ++t)
    ThPool[t].join();

  pcl::Feature<PointInT, PointOutT>::deinitCompute();

}


#define PCL_INSTANTIATE_NormalEstimationOMP(T,NT) template class PCL_EXPORTS pcl::NormalEstimationOMP<T,NT>;

#endif    // PCL_FEATURES_IMPL_NORMAL_3D_OMP_H_


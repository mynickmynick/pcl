/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
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
 */

#ifndef PCL_SEGMENTATION_IMPL_CONDITIONAL_EUCLIDEAN_CLUSTERING_HPP_
#define PCL_SEGMENTATION_IMPL_CONDITIONAL_EUCLIDEAN_CLUSTERING_HPP_

#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/search/organized.h> // for OrganizedNeighbor
#include <pcl/search/kdtree.h> // for KdTree
//#include <../../surface/include/pcl/surface/convex_hull.h>
#include <pcl/surface/convex_hull.h>

template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segment (pcl::IndicesClusters &clusters)
{
  // Prepare output (going to use push_back)
  clusters.clear ();
  if (extract_removed_clusters_)
  {
    small_clusters_->clear ();
    large_clusters_->clear ();
  }

  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty () || !condition_function_)
    return;

  // Initialize the search class
  if (!searcher_)
  {
    if (input_->isOrganized ())
      searcher_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
    else
      searcher_.reset (new pcl::search::KdTree<PointT> ());
  }
  searcher_->setInputCloud (input_, indices_);

  // Temp variables used by search class
  Indices nn_indices;
  std::vector<float> nn_distances;

  // Create a bool vector of processed point indices, and initialize it to false
  // Need to have it contain all possible points because radius search can not return indices into indices
  std::vector<bool> processed (input_->size (), false);

  // Process all points indexed by indices_
  for (const auto& iindex : (*indices_)) // iindex = input index
  {
    // Has this point been processed before?
    if (iindex == UNAVAILABLE || processed[iindex])
      continue;

    // Set up a new growing cluster
    Indices current_cluster;
    int cii = 0;  // cii = cluster indices iterator

    // Add the point to the cluster
    current_cluster.push_back (iindex);
    processed[iindex] = true;

    // Process the current cluster (it can be growing in size as it is being processed)
    while (cii < static_cast<int> (current_cluster.size ()))
    {
      // Search for neighbors around the current seed point of the current cluster
      if (searcher_->radiusSearch ((*input_)[current_cluster[cii]], cluster_tolerance_, nn_indices, nn_distances) < 1)
      {
        cii++;
        continue;
      }

      // Process the neighbors
      for (int nii = 1; nii < static_cast<int> (nn_indices.size ()); ++nii)  // nii = neighbor indices iterator
      {
        // Has this point been processed before?
        if (nn_indices[nii] == UNAVAILABLE || processed[nn_indices[nii]])
          continue;

        // Validate if condition holds
        if (condition_function_ ((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]))
        {
          // Add the point to the cluster
          current_cluster.push_back (nn_indices[nii]);
          processed[nn_indices[nii]] = true;
        }
      }
      cii++;
    }

    // If extracting removed clusters, all clusters need to be saved, otherwise only the ones within the given cluster size range
    if (extract_removed_clusters_ ||
        (static_cast<int> (current_cluster.size ()) >= min_cluster_size_ &&
         static_cast<int> (current_cluster.size ()) <= max_cluster_size_))
    {
      pcl::PointIndices pi;
      pi.header = input_->header;
      pi.indices.resize (current_cluster.size ());
      for (int ii = 0; ii < static_cast<int> (current_cluster.size ()); ++ii)  // ii = indices iterator
        pi.indices[ii] = current_cluster[ii];

      if (extract_removed_clusters_ && static_cast<int> (current_cluster.size ()) < min_cluster_size_)
        small_clusters_->push_back (pi);
      else if (extract_removed_clusters_ && static_cast<int> (current_cluster.size ()) > max_cluster_size_)
        large_clusters_->push_back (pi);
      else
        clusters.push_back (pi);
    }
  }

  deinitCompute ();
}

template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segment_ (pcl::IndicesClusters &clusters)
{
  // Prepare output (going to use push_back)
  clusters.clear ();
  if (extract_removed_clusters_)
  {
    small_clusters_->clear ();
    large_clusters_->clear ();
  }

  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty () || !condition_function_)
    return;

  // Initialize the search class
  if (!searcher_)
  {
    if (input_->isOrganized ())
      searcher_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
    else
      searcher_.reset (new pcl::search::KdTree<PointT> ());
  }
  searcher_->setInputCloud (input_, indices_);

  // Temp variables used by search class
  Indices nn_indices;
  std::vector<float> nn_distances;

  // Create a bool vector of processed point indices, and initialize it to false
  // Need to have it contain all possible points because radius search can not return indices into indices
  std::vector<bool> processed (input_->size (), false);

  // Process all points indexed by indices_
  for (const auto& iindex : (*indices_)) // iindex = input index
  {
    // Has this point been processed before?
    if (iindex == UNAVAILABLE || processed[iindex])
      continue;

    // Set up a new growing cluster
    Indices current_cluster;
    int cii = 0;  // cii = cluster indices iterator

    // Add the point to the cluster
    //std::cout << "NEW CLUSTER - - - - - - - - - - - - - - \n";
    current_cluster.push_back (iindex);

    cloud_cluster.push_back(PointCloudPtr(new pcl::PointCloud<PointT >) );
    //cloud_cluster_hull.push_back(PointCloudPtr(new pcl::PointCloud<PointT >) );
    PointCloudPtr tempHull1 = PointCloudPtr(new pcl::PointCloud<PointT >);
    //PointCloudPtr tempHull2 = PointCloudPtr(new pcl::PointCloud<PointT >);
    PointCloudPtr tempHull3 = PointCloudPtr(new pcl::PointCloud<PointT >);
    cloud_cluster.back()->push_back((*input_)[iindex]);
    tempHull1->push_back((*input_)[iindex]);
    tempHull3->push_back((*input_)[iindex]);
    int tempHull2Size = 1;
    processed[iindex] = true;

    // Process the current cluster (it can be growing in size as it is being processed)
    while (cii < static_cast<int> (current_cluster.size ()))
    {
      //std::cout << "Trying new center ------\n";
      // Search for neighbors around the current seed point of the current cluster
      if (searcher_->radiusSearch ((*input_)[current_cluster[cii]], cluster_tolerance_, nn_indices, nn_distances) < 1)
      {
        cii++;
        //std::cout << "Center skipped \n";
        continue;
      }

      // Process the neighbors
      //std::cout << "Processing neigbours " << static_cast<int> (nn_indices.size()) << std::endl;
      for (int nii = 1; nii < static_cast<int> (nn_indices.size ()); ++nii)  // nii = neighbor indices iterator
      {

        // Has this point been processed before?
        if (nn_indices[nii] == UNAVAILABLE || processed[nn_indices[nii]])
          continue;

        // Validate if condition holds
        if (condition_function_ ((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]))
        {
          float area, volume;
          if(cloud_cluster.back()->size()>10) //(tempHull1->size() > 4)
          {
            pcl::ConvexHull<PointT> chull;
            tempHull3->push_back((*input_)[nn_indices[nii]]);
            //std::cout <<"LOADING tempHull3 size "<< tempHull3->size() << std::endl;
            chull.setInputCloud(tempHull3);
            //chull.setDimension(3);
            chull.setComputeAreaVolume(true);
            PointCloudPtr tempHull2 = PointCloudPtr(new pcl::PointCloud<PointT >);
            int res=chull.reconstruct(*tempHull2);
            area = chull.getTotalArea();
            volume = chull.getTotalVolume();
            /*if(0)// (res)
            {
              //std::cout << "Loading full set!!!!\n";
              pcl::ConvexHull<PointT> chull2;
              chull2.setInputCloud(cloud_cluster.back());
              //chull2.setDimension(3);
              chull2.setComputeAreaVolume(true);
              res=chull2.reconstruct(*tempHull2);
              area = chull2.getTotalArea();
              volume = chull2.getTotalVolume();
            }
            */
            if (!res &&
              216 * volume * volume <=
              UnflatnessThreshold * (area * area * area))
            {
                /*
                tempHull1->resize(0);
                tempHull3->resize(0);
                tempHull1->width = tempHull3->width=tempHull2->width;
                tempHull1->height = tempHull3->height=tempHull2->height;
                tempHull1->is_dense = tempHull3->is_dense=tempHull2->is_dense;
                tempHull1->points.resize(tempHull2->size());
                tempHull3->points.resize(tempHull2->size());
                for (int i = 0; i < tempHull2->size(); ++i)
                {
                  tempHull1->points[i] = tempHull3->points[i] = tempHull2->points[i];
                }
                */
                if (
                  //tempHull2Size <= tempHull2->size() &&
                  !res)
                {
                  tempHull1 = tempHull2;
                  tempHull3 = tempHull2;
                  tempHull2Size = tempHull2->size();
                  //std::cout <<"tempHull2 size "<< tempHull2->size() << std::endl;
                }
                else
                {
                  tempHull1 = tempHull3;
                }
                // Add the point to the cluster
                current_cluster.push_back(nn_indices[nii]);
                cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
                processed[nn_indices[nii]] = true;
                /*std::cout << "new------------Inserted Point -----\n";
                std::cout <<"tempHull size "<< tempHull1->size() << std::endl;
                std::cout <<"tempHull3 size "<< tempHull3->size() << std::endl;
                std::cout <<"cloud_cluster.back.size:: "<< cloud_cluster.back()->size() << std::endl;
                for (auto p: tempHull1->points)
                  std::cout <<" -- P: "<< p.x<<" : "<< p.y<<" : "<< p.z;
                std::cout << std::endl;
                getchar();*/

            }
            else
            {
              /*
              tempHull3->resize(0);
              tempHull3->width=tempHull1->width;
               tempHull3->height=tempHull1->height;
              tempHull3->is_dense=tempHull1->is_dense;
              tempHull3->points.resize(tempHull1->size());
              for (int i = 0; i < tempHull1->size(); ++i)
              {
                tempHull3->points[i] = tempHull1->points[i];
              }
              */
              tempHull3 = tempHull1;
              /*std::cout << "Rejecting Point -----\n";
              std::cout <<"tempHull1 size "<< tempHull1->size() << std::endl;
              std::cout <<"tempHull3 size "<< tempHull3->size() << std::endl;
              std::cout <<"cloud_cluster.back.size:: "<< cloud_cluster.back()->size() << std::endl;
              for (auto p: tempHull1->points)
                std::cout <<" -- P: "<< p.x<<" : "<< p.y<<" : "<< p.z;
              std::cout << std::endl;
              getchar();*/

            }
          }
          else
          {
            // Add the point to the cluster
            current_cluster.push_back(nn_indices[nii]);
            cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
            tempHull3->push_back((*input_)[nn_indices[nii]]);
            tempHull1->push_back((*input_)[nn_indices[nii]]);
            processed[nn_indices[nii]] = true;
            /*std::cout << "------------Inserted Point at cluster start -----\n";
            std::cout <<"tempHull1 size "<< tempHull1->size() << std::endl;
            std::cout <<"tempHull3 size "<< tempHull3->size() << std::endl;
            std::cout <<"cloud_cluster.back.size:: "<< cloud_cluster.back()->size() << std::endl;
            for (auto p: tempHull1->points)
              std::cout <<" -- P: "<< p.x<<" : "<< p.y<<" : "<< p.z;
            std::cout << std::endl;
            getchar();*/
          }
        }
      }
      cii++;
    }

    // If extracting removed clusters, all clusters need to be saved, otherwise only the ones within the given cluster size range
    if (extract_removed_clusters_ ||
      (static_cast<int> (current_cluster.size ()) >= min_cluster_size_ &&
        static_cast<int> (current_cluster.size ()) <= max_cluster_size_))
    {
      pcl::PointIndices pi;
      pi.header = input_->header;
      pi.indices.resize (current_cluster.size ());
      for (int ii = 0; ii < static_cast<int> (current_cluster.size ()); ++ii)  // ii = indices iterator
        pi.indices[ii] = current_cluster[ii];

      if (extract_removed_clusters_ && static_cast<int> (current_cluster.size ()) < min_cluster_size_)
        small_clusters_->push_back (pi);
      else if (extract_removed_clusters_ && static_cast<int> (current_cluster.size ()) > max_cluster_size_)
        large_clusters_->push_back (pi);
      else
        clusters.push_back (pi);
    }
  }

  deinitCompute ();
}


#define PCL_INSTANTIATE_ConditionalEuclideanClustering(T) template class PCL_EXPORTS pcl::ConditionalEuclideanClustering<T>;

#endif  // PCL_SEGMENTATION_IMPL_CONDITIONAL_EUCLIDEAN_CLUSTERING_HPP_


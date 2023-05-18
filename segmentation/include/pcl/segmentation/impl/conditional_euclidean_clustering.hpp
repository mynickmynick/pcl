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
#include <pcl/surface/convex_hull.h>
#include <pcl/common/centroid.h>
#include <vector>
#include <algorithm>
#include <mutex>




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
pcl::ConditionalEuclideanClustering<PointT>::segment_ByConvexHull (pcl::IndicesClusters &clusters)
{
  bool condition = true, conditionDisabled=(!condition_function_);
  // Prepare output (going to use push_back)
  clusters.clear ();
  if (extract_removed_clusters_)
  {
    small_clusters_->clear ();
    large_clusters_->clear ();
  }

  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty ())
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
    PointCloudPtr tempHull2 = PointCloudPtr(new pcl::PointCloud<PointT >);
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
        if (!conditionDisabled)
          condition = condition_function_((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]);
        if (condition)
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
            //PointCloudPtr tempHull2 = PointCloudPtr(new pcl::PointCloud<PointT >);
            //std::cout <<"BEFORE--:"<< *tempHull2 << "\n";
            //std::cout << tempHull2->width <<  " "<<tempHull2->height<<" "<<tempHull2->is_dense<<    "\n";

            int res = chull.reconstruct(tempHull2);//(*tempHull2);
            //std::cout <<"AFTER--:"<< *tempHull2 << "\n";
            //std::cout << tempHull2->width <<  " "<<tempHull2->height<<" "<<tempHull2->is_dense<<    "\n";
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
                  //std::cout << "------------tempHull2 size and res " << tempHull2->size() << " " <<res << std::endl;
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
              //std::cout << "Rejecting Point -----\n";
              //std::cout <<"tempHull1 size "<< tempHull1->size() << std::endl;
              //std::cout <<"tempHull3 size "<< tempHull3->size() << std::endl;
              //std::cout <<"cloud_cluster.back.size:: "<< cloud_cluster.back()->size() << std::endl;
              //for (auto p: tempHull1->points)
              //  std::cout <<" -- P: "<< p.x<<" : "<< p.y<<" : "<< p.z;
              //std::cout << std::endl;
              //getchar();

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
            std::cout <<"tempHull1 size "<< tempHull1->size() << std::endl;
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

template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segment_ByOBB (pcl::IndicesClusters &clusters,
  size_t OBB_UpdatePeriod_SamplesNr, size_t OBB_CalculationStart_UpdatePeriodNr)
{
  if (!OBB_UpdatePeriod_SamplesNr)
    OBB_UpdatePeriod_SamplesNr = 1;

  bool condition = true, conditionDisabled=(!condition_function_);
  // Prepare output (going to use push_back)
  clusters.clear ();


  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty ())
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
  std::vector<int> processed (input_->size (), false);

  int clusterIndex = 1;
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

    cloud_cluster.push_back(PointCloudPtr(new pcl::PointCloud<PointT >) );
    //cloud_cluster_hull.push_back(PointCloudPtr(new pcl::PointCloud<PointT >) );

    Eigen::Matrix<float, 3, 1> centroid;
    Eigen::Matrix<float, 3, 3>  covariance_matrix ;
    Eigen::Matrix<float, 3, 1> obb_center;
    Eigen::Matrix<float, 3, 1> obb_dimensions;
    Eigen::Matrix<float, 3, 3> obb_rotational_matrix;
    unsigned int oldSize = 0;
    size_t point_count = 0;
    Eigen::Matrix<float, 3, 1> major_axis;
    Eigen::Matrix<float, 3, 1> middle_axis;
    Eigen::Matrix<float, 3, 1> minor_axis;


    cloud_cluster.back()->push_back((*input_)[iindex]);

    processed[iindex] = clusterIndex;

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
        if (!conditionDisabled)
          condition = condition_function_((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]);
        if (condition)
        {

          if(cloud_cluster.back()->size()>OBB_UpdatePeriod_SamplesNr*OBB_CalculationStart_UpdatePeriodNr)//50
          {
            if (cloud_cluster.back()->size() % OBB_UpdatePeriod_SamplesNr == 1)//this period must be a submultiple of the previous period //25
            {
              Eigen::Matrix<float, 3, 1> temp_centroid=centroid;
              Eigen::Matrix<float, 3, 3> temp_covariance_matrix=covariance_matrix;
              Eigen::Matrix<float, 3, 1> temp_obb_center=obb_center;
              Eigen::Matrix<float, 3, 1> temp_obb_dimensions=obb_dimensions;
              Eigen::Matrix<float, 3, 3> temp_obb_rotational_matrix=obb_rotational_matrix;
              unsigned int temp_oldSize = oldSize;
              size_t temp_point_count=point_count;

              updateCentroidAndOBB(*(cloud_cluster.back()),
                temp_centroid,
                temp_covariance_matrix,
                temp_obb_center,
                temp_obb_dimensions,
                temp_obb_rotational_matrix,
                temp_oldSize,
                temp_point_count);

              //volume = temp_obb_dimensions[0] * temp_obb_dimensions[1] * temp_obb_dimensions[2];
              //area = temp_obb_dimensions[0] * temp_obb_dimensions[1];

              if (//flatness condition
                //volume * volume <=
                //UnflatnessThreshold * (area * area * area)
                temp_obb_dimensions[2] * temp_obb_dimensions[2] <=
                UnflatnessThreshold * (temp_obb_dimensions[0] * temp_obb_dimensions[1])
                )//unflatness: [0,1] 0:perfectly flat, 1:cube
              {

                // Add the point to the cluster
                current_cluster.push_back(nn_indices[nii]);
                cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
                processed[nn_indices[nii]] = clusterIndex;

                centroid=temp_centroid;
                covariance_matrix=temp_covariance_matrix;
                obb_center=temp_obb_center;
                obb_dimensions=temp_obb_dimensions;
                obb_rotational_matrix=temp_obb_rotational_matrix;
                oldSize =  temp_oldSize;
                point_count = temp_point_count;

                major_axis= obb_rotational_matrix.col(0);
                middle_axis= obb_rotational_matrix.col(1);
                minor_axis= obb_rotational_matrix.col(2);

              }

            }
            else
            {
              float xd = (*input_)[nn_indices[nii]].x - centroid[0],
                yd = (*input_)[nn_indices[nii]].y - centroid[1],
                zd = (*input_)[nn_indices[nii]].z - centroid[2];

              float x = std::abs(xd * major_axis(0) + yd * major_axis(1) + zd * major_axis(2));
              float y = std::abs(xd * middle_axis(0) + yd * middle_axis(1) + zd * middle_axis(2));
              float z = std::abs(xd * minor_axis(0) + yd * minor_axis(1) + zd * minor_axis(2));

                if (//flatness condition
                  (z<obb_dimensions[2]*0.75)||//(z<obb_dimensions[2]*0.5)||
                  (z * z <=
                  1.5*UnflatnessThreshold * (x * y))//UnflatnessThreshold * (x * y))
                  )//unflatness: [0,1] 0:perfectly flat, 1:cube
                {
                  // Add the point to the cluster
                  current_cluster.push_back(nn_indices[nii]);
                  cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
                  processed[nn_indices[nii]] = clusterIndex;
                }
              

            }




          }
          else
          {
            // Add the point to the cluster
            current_cluster.push_back(nn_indices[nii]);
            cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);

            processed[nn_indices[nii]] = clusterIndex;

          }
        }
      }
      cii++;
    }

    //  clusters need to be saved only the ones within the given cluster size range
    if (
      (static_cast<int> (current_cluster.size ()) >= min_cluster_size_ &&
        static_cast<int> (current_cluster.size ()) <= max_cluster_size_))
    {
      pcl::PointIndices pi;
      pi.header = input_->header;
      pi.indices.resize (current_cluster.size ());
      for (int ii = 0; ii < static_cast<int> (current_cluster.size ()); ++ii)  // ii = indices iterator
        pi.indices[ii] = current_cluster[ii];

      clusters.push_back (pi);
    }


  }

  deinitCompute ();
}


//every time a thread writes on an area of memory being read by another thread not only potentially mutex-blocks it but also invalidates its cache so making it waste
//both processing and memory time
template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segmentThreadOld(
  //SearcherPtr& searcher_,
    std::mutex & clusters_mutex,
  std::vector<size_t> & processed,
  std::vector<std::shared_mutex> & processed_mutex,
  std::unordered_set<PairS> & connections_out,
  size_t i0, size_t i1, size_t threadNumber

)
{
  std::unordered_set<PairS> connections;
  shared_ptr<pcl::PointCloud<PointT>> input(new pcl::PointCloud<PointT>);
  shared_ptr<Indices> indices=make_shared<Indices>();
  connections.clear();

  input->points.resize( this->input_->points.size());
  input->width = this->input_->width;
  input->height = this->input_->height;
  input->is_dense = this->input_->is_dense;
  for (size_t i=0;i< this->input_->points.size();++i)
  {
    input->points[i] = this->input_->points[i];
  }

  indices->resize(this->indices_->size());
  for (size_t i=0;i< this->indices_->size();++i)
  {
    (*indices)[i] = (*(this->indices_))[i];
  }


  SearcherPtr searcher_;
  // Initialize the search class
  if (!searcher_)
  {
    if (input->isOrganized ())
      searcher_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
    else
      searcher_.reset (new pcl::search::KdTree<PointT> ());
    //searcher_.reset (new pcl::search::FlannSearch<PointT> ());
  }
  searcher_->setInputCloud (input, indices);

  //std::map<size_t, shared_ptr<pcl::PointIndices>> clusterRecords;
  size_t local_current_cluster_index = 1;//[1..]

  // Temp variables used by search class
  Indices nn_indices;
  std::vector<float> nn_distances;

  // Process all points indexed by indices_
  // the following map may seem weird but it is more fair in distribution of work load because the algorithm proceeds by connection both bacward and forward on indexes
  for(size_t j=i0;j<i1;++j)
  {
    size_t i = j + (i1 - i0) / 2;
    if (i >= i1)
      i = i0 + (i - i1);
    auto iindex = (*indices)[i];
    {
      local_current_cluster_index=iindex;//local_current_cluster_index= index of first point added to the cluster
    }

    // Set up a new growing cluster
    shared_ptr<pcl::PointIndices> pi=make_shared<pcl::PointIndices>();
    pi->header = input->header;
    Indices & current_cluster=pi->indices;
    int cii = 0;  // cii = cluster indices iterator

    size_t processed_ = 0;
    {
      std::unique_lock<std::shared_mutex> ulock(processed_mutex[iindex]);
      processed_ = processed[iindex];

      // Has this point been processed before?
      if (iindex == UNAVAILABLE || processed_)
        continue;

      // Add the FIRST point to the cluster
      processed[iindex] = local_current_cluster_index;
    }
    current_cluster.push_back (iindex);

    // Process the current cluster (it can be growing in size as it is being processed)
    while (cii < static_cast<int> (current_cluster.size ()))
    {
      // Search for neighbors around the current seed point of the current cluster
      if (searcher_->radiusSearch ((*input)[current_cluster[cii]], cluster_tolerance_, nn_indices, nn_distances) < 1)
      {
        cii++;
        continue;
      }

      // Process the neighbors
      for (int nii = 1; nii < static_cast<int> (nn_indices.size ()); ++nii)  // nii = neighbor indices iterator
      {
        // Has this point been processed before?
        if (nn_indices[nii] == UNAVAILABLE )
          continue;

        size_t processed_ = 0;
        {
          std::shared_lock<std::shared_mutex> slock(processed_mutex[nn_indices[nii]]);
          processed_ = processed[nn_indices[nii]];
        }

        // Has this point been processed before?
        if (processed_ == local_current_cluster_index)
          continue;

        // Validate if condition holds
        if (condition_function_ ((*input)[current_cluster[cii]], (*input)[nn_indices[nii]], nn_distances[nii]))
        {
          if (processed_)
          {
            PairS p;
            p.first = local_current_cluster_index; p.second= processed_;
            //if (!connections.count(p)) redundant for sets
              connections.insert(p);
          }
          else
          {
              {
              std::unique_lock<std::shared_mutex> ulock(processed_mutex[nn_indices[nii]]);
                //I have to test it again cause it might have been processed in the meantime
                processed_ = processed[nn_indices[nii]];

                if (processed_)
                {
                  PairS p;
                  p.first = local_current_cluster_index; p.second= processed_;
                  //if (!connections.count(p))
                    connections.insert(p);
                }
                else
                {// Add the point to the cluster
                  current_cluster.push_back (nn_indices[nii]);
                  processed[nn_indices[nii]] = local_current_cluster_index;
                }
              }
          }


        }
      }
      cii++;
    }


    

      {
        //const std::lock_guard<std::mutex> lock(clusters_mutex);
        clusterRecordsGlob[local_current_cluster_index]=pi;
      }
      {
          //std::unique_lock<std::shared_mutex> ul(connections_mutex);
          //if (local_current_cluster_index> max_cluster_index)
          //  max_cluster_index = local_current_cluster_index;
          //local_current_cluster_index=++current_cluster_index;
      }
    


  }

  connections_out.clear();
  connections_out = connections;

}



template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segmentThread(
  SearcherPtr& searcher_,
    std::mutex & clusters_mutex,
  std::vector<size_t> & processed,
  std::vector<std::shared_mutex> & processed_mutex,
  size_t i0, size_t i1

)
//this is based on a total separation of indexes in write mode between the threads
{
  //std::map<size_t, shared_ptr<pcl::PointIndices>> clusterRecords;
  size_t local_current_cluster_index = 1;//[1..]
  {
      std::unique_lock<std::shared_mutex> ul(connections_mutex);
      local_current_cluster_index=++current_cluster_index;
  }

  std::unordered_set<PairS> local_connections;

  // Temp variables used by search class
  Indices nn_indices;
  std::vector<float> nn_distances;

  std::unordered_set<index_t> indexSet;
  for (size_t i = i0; i < i1; ++i)//record the index set for later
  {
    indexSet.insert((*indices_)[i]);
  }

  // Process all points indexed by indices_
  for(size_t i=i0;i<i1;++i)
  {
    auto iindex = (*indices_)[i];

    // Set up a new growing cluster
    shared_ptr<pcl::PointIndices> pi=make_shared<pcl::PointIndices>();
    pi->header = input_->header;
    Indices & current_cluster=pi->indices;
    int cii = 0;  // cii = cluster indices iterator

    size_t processed_ = 0;
    {
      std::shared_lock<std::shared_mutex> slock(processed_mutex[iindex]);
      processed_ = processed[iindex];
    }

    // Has this point been processed before?
    if (iindex == UNAVAILABLE || processed_)
      continue;

    // Add the FIRST point to the cluster
    current_cluster.push_back (iindex);
    {
      std::unique_lock<std::shared_mutex> ulock(processed_mutex[iindex]);
      processed[iindex] = local_current_cluster_index;
    }



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
        if (nn_indices[nii] == UNAVAILABLE )
          continue;

        size_t processed_ = 0;
        {
          std::shared_lock<std::shared_mutex> slock(processed_mutex[nn_indices[nii]]);
          processed_ = processed[nn_indices[nii]];
        }

        // Has this point been processed before?
        if (processed_ == local_current_cluster_index)
          continue;

        // Validate if condition holds
        if (condition_function_ ((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]))
        {
          if (processed_)//at this point it has to be !=local_current_cluster_index
          {
            PairS p;
            p.first = local_current_cluster_index; p.second= processed_;
            if (!local_connections.count(p))
            {
              local_connections.insert(p);
              {
                std::unique_lock<std::shared_mutex> ul(connections_mutex);
                gconnections.insert(p);//the two growing clusters will have to be connected
              }
            }


          }
          else
          {
              if (indexSet.count(nn_indices[nii]))//if it belongs to my assigned set I will fully process it, otherwise just record the possible connection
              {//if it belongs to my assigned set only me can process it so no need to test it again, it has to be not processed at this point


                current_cluster.push_back (nn_indices[nii]);
                {// Add the point to the cluster
                  std::unique_lock<std::shared_mutex> ulock(processed_mutex[iindex]);
                  processed[nn_indices[nii]] = local_current_cluster_index;
                }
              }
              //else
                //I would have to test again processed cause it might have been processed in the meantime
                //BUT it is not necessary cause the connection will be detected on the other side
                //and this is not a point assigned to me so I have nothing left to do
          }


        }
      }
      cii++;
    }


    

      {
        //const std::lock_guard<std::mutex> lock(clusters_mutex);
        clusterRecordsGlob[local_current_cluster_index]=pi;
      }
      {
          std::unique_lock<std::shared_mutex> ul(connections_mutex);
          if (local_current_cluster_index> max_cluster_index)
            max_cluster_index = local_current_cluster_index;
          local_current_cluster_index=++current_cluster_index;
      }
    


  }
      {
        //const std::lock_guard<std::mutex> lock(clusters_mutex);
        //for (auto& c : clusterRecords)
        //  clusterRecordsGlob[c.first] = c.second;
      }
}



template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segmentMT (pcl::IndicesClusters &clusters, const size_t threadNumber)
{
  // Prepare output (going to use push_back)
  clusters.clear ();
  std::mutex clusters_mutex;
  current_cluster_index = 0;

  clusterRecordsGlob.clear();
  clusterRecordsGlob.resize(input_->size());
  for (auto& c : clusterRecordsGlob) c.reset();

  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty () || !condition_function_)
    return;

  // Create a bool vector of processed point indices, and initialize it to false
  // Need to have it contain all possible points because radius search can not return indices into indices
  std::vector<size_t> processed (input_->size (), 0);
  std::vector<std::shared_mutex> processed_mutex(input_->size ());

  size_t chunk = indices_->size() / threadNumber;
  std::vector<std::thread> ThPool;
  std::shared_ptr<std::unordered_set<PairS>> connections[32];
  size_t i0 = 0;
  size_t i1 = chunk;
  for (size_t t = 0; t < threadNumber; ++t)
  {
    if (t == threadNumber - 1)
      i1 = indices_->size();
    connections[t] = std::make_shared<std::unordered_set<PairS>>();
    connections[t]->clear();

    ThPool.push_back( std::move( std::thread(&pcl::ConditionalEuclideanClustering<PointT>::segmentThreadOld,this,//pcl::ConditionalEuclideanClustering::segmentThread<PointT>,
      std::ref(clusters_mutex),
      std::ref(processed),
      std::ref(processed_mutex),
      std::ref(*(connections[t])),
      i0, i1, t
    )));
    i0 += chunk;
    i1 += chunk;
  }

  for (size_t t = 0; t < threadNumber; ++t)
    ThPool[t].join();
  
  std::vector<std::set<size_t>> partition;//only partial partitions cause it doesn't include singletons
  for (size_t t=0;t<threadNumber;++t)
    for (auto& conn : (*(connections[t])))
  {
    bool found = false;
    std::vector<size_t> lastFound;
    for (size_t i= 0;i<partition.size();++i)
    {
     
      if ( partition[i].count(conn.first ))
      {
        found = true;
        partition[i].insert(conn.second);
        lastFound.push_back(i);
        continue;
      }
      if ( partition[i].count(conn.second))
      {
        found = true;
        partition[i].insert(conn.first);
        lastFound.push_back(i);
        continue;
      }
    }
    while (lastFound.size()>1)
    {
      partition[lastFound[lastFound.size()-2]].merge(partition[lastFound[lastFound.size()-1]]);
      partition[lastFound[lastFound.size() - 1]].clear();
      lastFound.pop_back();
    }

    if (!found)
    {
      std::set<size_t> s;
      s.insert(conn.first);
      s.insert(conn.second);
      partition.push_back(s);
    }
  }
  for (auto & p: partition)
    if (p.size())
  {
    size_t ss = 0;
    for (auto& c : p)
    {
      if (clusterRecordsGlob[c])
      {
        ss += clusterRecordsGlob[c]->indices.size();
      }
    }
    if (
      static_cast<int> (ss) >= min_cluster_size_ &&
      static_cast<int> (ss) <= max_cluster_size_)
    {
        pcl::PointIndices pi;
        pi.header = input_->header;
        pi.indices.resize (ss);

        auto pii = pi.indices.begin();
        
        for (auto& c : p)
        {

          if (clusterRecordsGlob[c])
          {
            pii=std::copy(clusterRecordsGlob[c]->indices.begin(), clusterRecordsGlob[c]->indices.end(), pii);
            clusterRecordsGlob[c].reset();
          }

        }
        clusters.push_back (pi);
    }
    else
      for (auto& c : p)
          clusterRecordsGlob[c].reset();


  }
  for (auto& c : clusterRecordsGlob)
  //for (size_t i=0;i<max_cluster_index;++i)
  {
    //auto c = clusterRecordsGlob[i];
    if(c)
    if (
      static_cast<int> (c->indices.size()) >= min_cluster_size_ &&
      static_cast<int> (c->indices.size()) <= max_cluster_size_)
    {
      pcl::PointIndices pi;
      pi.header = input_->header;
      pi.indices.resize (c->indices.size());

      std::copy(c->indices.begin(), c->indices.end(), pi.indices.begin());
      clusters.push_back (pi);
    }
  }


  deinitCompute ();
}




template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segment_ByOBBThread(
  SearcherPtr& searcher_,
  pcl::IndicesClusters &clusters,
    std::mutex & clusters_mutex,
  std::vector<std::set<size_t>> & processed,
  std::vector<std::shared_mutex> & processed_mutex,
  size_t i0, size_t i1,
  bool record_connections//only prepared not used yet (false)

)
{
  std::map<size_t, shared_ptr<pcl::PointIndices>> clusterRecords;
  std::vector<PointCloudPtr> local_cloud_cluster;
  bool condition = true, conditionDisabled=(!condition_function_);

  size_t local_current_cluster_index = 1;//[1..]

  {
      std::unique_lock<std::shared_mutex> ul(connections_mutex);
      local_current_cluster_index=++current_cluster_index;
  }


  std::unordered_set<PairS> local_connections;

  // Temp variables used by search class
  Indices nn_indices;
  std::vector<float> nn_distances;

  // Process all points indexed by indices_
  for(size_t i=i0;i<i1;++i)
  {
    auto iindex = (*indices_)[i];

    std::set<size_t> processed_;
    {
      std::shared_lock<std::shared_mutex> slock(processed_mutex[iindex]);
      processed_ = processed[iindex];
    }

    // Has this point been processed before?
    if (iindex == UNAVAILABLE || processed_.size()>0)
      continue;

    // Set up a new growing cluster
    shared_ptr<pcl::PointIndices> pi=make_shared<pcl::PointIndices>();
    pi->header = input_->header;
    Indices & current_cluster=pi->indices;
    int cii = 0;  // cii = cluster indices iterator

    // Add the FIRST point to the cluster
    current_cluster.push_back (iindex);
    {
      std::unique_lock<std::shared_mutex> ulock(processed_mutex[iindex]);
      processed[iindex].insert(local_current_cluster_index);
    }
    Eigen::Matrix<float, 3, 1> centroid;
    Eigen::Matrix<float, 3, 3>  covariance_matrix ;
    Eigen::Matrix<float, 3, 1> obb_center;
    Eigen::Matrix<float, 3, 1> obb_dimensions;
    Eigen::Matrix<float, 3, 3> obb_rotational_matrix;
    unsigned int oldSize = 0;
    size_t point_count = 0;
    Eigen::Matrix<float, 3, 1> major_axis;
    Eigen::Matrix<float, 3, 1> middle_axis;
    Eigen::Matrix<float, 3, 1> minor_axis;
    local_cloud_cluster.push_back(PointCloudPtr(new pcl::PointCloud<PointT >) );
    local_cloud_cluster.back()->push_back((*input_)[iindex]);
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
        if (nn_indices[nii] == UNAVAILABLE )
          continue;

        std::set<size_t> processed_;
        {
          std::shared_lock<std::shared_mutex> slock(processed_mutex[nn_indices[nii]]);
          processed_ = processed[nn_indices[nii]];
        }

        // Has this point been processed before?
        if (processed_.count(local_current_cluster_index))
          continue;

        // Validate if condition holds
        if (!conditionDisabled)
          condition = condition_function_((*input_)[current_cluster[cii]], (*input_)[nn_indices[nii]], nn_distances[nii]);

        if (condition)
        {

          if (local_cloud_cluster.back()->size() > 50)//40
          {
            if (local_cloud_cluster.back()->size() % 50 == 1)//this period must be a submultiple of the previous period //20
            {
              Eigen::Matrix<float, 3, 1> temp_centroid = centroid;
              Eigen::Matrix<float, 3, 3> temp_covariance_matrix = covariance_matrix;
              Eigen::Matrix<float, 3, 1> temp_obb_center = obb_center;
              Eigen::Matrix<float, 3, 1> temp_obb_dimensions = obb_dimensions;
              Eigen::Matrix<float, 3, 3> temp_obb_rotational_matrix = obb_rotational_matrix;
              unsigned int temp_oldSize = oldSize;
              size_t temp_point_count = point_count;

              updateCentroidAndOBB(*(local_cloud_cluster.back()),
                temp_centroid,
                temp_covariance_matrix,
                temp_obb_center,
                temp_obb_dimensions,
                temp_obb_rotational_matrix,
                temp_oldSize, temp_point_count);

              //volume = temp_obb_dimensions[0] * temp_obb_dimensions[1] * temp_obb_dimensions[2];
              //area = temp_obb_dimensions[0] * temp_obb_dimensions[1];
              if (//flatness condition
                //volume * volume <=
                //UnflatnessThreshold * (area * area * area)
                temp_obb_dimensions[2] * temp_obb_dimensions[2] <=
                UnflatnessThreshold * (temp_obb_dimensions[0] * temp_obb_dimensions[1])
                )//unflatness: [0,1] 0:perfectly flat, 1:cube
              {

                {// Add the point to the cluster
                  if (record_connections && processed_.size() > 0)
                  {
                    for (auto& a : processed_)
                    {
                      if (a != local_current_cluster_index)
                      {
                        PairS p;
                        p.first = local_current_cluster_index; p.second = a;
                        if (!local_connections.count(p))
                        {
                          local_connections.insert(p);
                          {
                            std::unique_lock<std::shared_mutex> ul(connections_mutex);
                            gconnections.insert(p);//the two growing clusters will have to be connected
                          }
                        }
                      }
                    }


                  }
                  //else
                  //{
                    // Add the point to the cluster anyway (if true the if above, it will be an intersection)
                  current_cluster.push_back(nn_indices[nii]);
                  local_cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
                  {
                    std::unique_lock<std::shared_mutex> ulock(processed_mutex[nn_indices[nii]]);
                    processed[nn_indices[nii]].insert(local_current_cluster_index);
                  }
                  //}

                }

                centroid = temp_centroid;
                covariance_matrix = temp_covariance_matrix;
                obb_center = temp_obb_center;
                obb_dimensions = temp_obb_dimensions;
                obb_rotational_matrix = temp_obb_rotational_matrix;
                oldSize = temp_oldSize;
                point_count = temp_point_count;

                major_axis = obb_rotational_matrix.col(0);
                middle_axis = obb_rotational_matrix.col(1);
                minor_axis = obb_rotational_matrix.col(2);

              }

            }
            else
            {
              float xd = (*input_)[nn_indices[nii]].x - centroid[0],
                yd = (*input_)[nn_indices[nii]].y - centroid[1],
                zd = (*input_)[nn_indices[nii]].z - centroid[2];

              float x = std::abs(xd * major_axis(0) + yd * major_axis(1) + zd * major_axis(2));
              float y = std::abs(xd * middle_axis(0) + yd * middle_axis(1) + zd * middle_axis(2));
              float z = std::abs(xd * minor_axis(0) + yd * minor_axis(1) + zd * minor_axis(2));

              if (//flatness condition
                (z < obb_dimensions[2] * 0.5) ||
                (z * z <=
                  UnflatnessThreshold * (x * y))
                )//unflatness: [0,1] 0:perfectly flat, 1:cube
              {// Add the point to the cluster
                if (record_connections && processed_.size() > 0)
                {
                  for (auto& a : processed_)
                  {
                    if (a != local_current_cluster_index)
                    {
                      PairS p;
                      p.first = local_current_cluster_index; p.second = a;
                      if (!local_connections.count(p))
                      {
                        local_connections.insert(p);
                        {
                          std::unique_lock<std::shared_mutex> ul(connections_mutex);
                          gconnections.insert(p);//the two growing clusters will have to be connected
                        }
                      }
                    }
                  }


                }
                //else
                //{
                  // Add the point to the cluster anyway (if true the if above, it will be an intersection)
                current_cluster.push_back(nn_indices[nii]);
                local_cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
                {
                  std::unique_lock<std::shared_mutex> ulock(processed_mutex[nn_indices[nii]]);
                  processed[nn_indices[nii]].insert(local_current_cluster_index);
                }
                //}

              }


            }




          }
          else
          {// Add the point to the cluster

            if (record_connections && processed_.size() > 0)
            {
              for (auto& a : processed_)
              {
                if (a != local_current_cluster_index)
                {
                  PairS p;
                  p.first = local_current_cluster_index; p.second = a;
                  if (!local_connections.count(p))
                  {
                    local_connections.insert(p);
                    {
                      std::unique_lock<std::shared_mutex> ul(connections_mutex);
                      gconnections.insert(p);//the two growing clusters will have to be connected
                    }
                  }
                }
              }


            }
            //else
            //{
              // Add the point to the cluster anyway (if true the if above, it will be an intersection)
            current_cluster.push_back(nn_indices[nii]);
            local_cloud_cluster.back()->push_back((*input_)[nn_indices[nii]]);
            {
              std::unique_lock<std::shared_mutex> ulock(processed_mutex[nn_indices[nii]]);
              processed[nn_indices[nii]].insert(local_current_cluster_index);
            }
            //}

          }
        }

      }
      cii++;
    }

        //  clusters need to be saved only the ones within the given cluster size range
    if (record_connections ||
      (static_cast<int> (current_cluster.size ()) >= min_cluster_size_ &&
        static_cast<int> (current_cluster.size ()) <= max_cluster_size_))
      {

        {
          
          if (record_connections)
            clusterRecords[local_current_cluster_index] = pi;
          else
          {
            const std::lock_guard<std::mutex> lock(clusters_mutex);
            clusters.push_back(*pi);
          }
        }
        {
          std::unique_lock<std::shared_mutex> ul(connections_mutex);
          if (local_current_cluster_index> max_cluster_index)
            max_cluster_index = local_current_cluster_index;
          local_current_cluster_index = ++current_cluster_index;
        }

      }


  }
  if (record_connections)
      {
        //const std::lock_guard<std::mutex> lock(clusters_mutex);
        for (auto& c : clusterRecords)
          clusterRecordsGlob[c.first] = c.second;
      }
}




template<typename PointT> void
pcl::ConditionalEuclideanClustering<PointT>::segment_ByOBBMT (pcl::IndicesClusters &clusters, const size_t threadNumber)
{

  // Prepare output (going to use push_back)
  clusters.clear ();
  clusterRecordsGlob.clear();
  clusterRecordsGlob.resize(input_->size());
  for (auto& c : clusterRecordsGlob) c.reset();
  std::mutex clusters_mutex;
  current_cluster_index = 0;

  // Validity checks
  if (!initCompute () || input_->points.empty () || indices_->empty ())
    return;


  SearcherPtr searcher_;
  // Initialize the search class
  if (!searcher_)
  {
    if (input_->isOrganized ())
      searcher_.reset (new pcl::search::OrganizedNeighbor<PointT> ());
    else
      searcher_.reset (new pcl::search::KdTree<PointT> ());
  }
  searcher_->setInputCloud (input_, indices_);


  // Create a bool vector of processed point indices, and initialize it to false
  // Need to have it contain all possible points because radius search can not return indices into indices
  std::vector<std::set<size_t>> processed (input_->size ());
  std::vector<std::shared_mutex> processed_mutex(input_->size ());

  size_t chunk = indices_->size() / threadNumber;
  std::vector<std::thread> ThPool;
  size_t i0 = 0;
  size_t i1 = chunk;
  for (size_t t = 0; t < threadNumber; ++t)
  {
    if (t == threadNumber - 1)
      i1 = indices_->size();

    ThPool.push_back( std::move( std::thread(&pcl::ConditionalEuclideanClustering<PointT>::segment_ByOBBThread,this,//pcl::ConditionalEuclideanClustering::segmentThread<PointT>,
      std::ref(searcher_),
      std::ref(clusters),
      std::ref(clusters_mutex),
      std::ref(processed),
      std::ref(processed_mutex),
      i0, i1, false
    )));
    i0 += chunk;
    i1 += chunk;
  }

  for (size_t t = 0; t < threadNumber; ++t)
    ThPool[t].join();

  std::vector<std::set<size_t>> partition;//only partial partitions cause it doesn't include singletons
  for (auto& conn : gconnections)
  {
    bool found = false;
    for (auto& s : partition)
    {
      if (s.count(conn.first))
      {
        found = true;
        s.insert(conn.second);
        break;
      }
      if (s.count(conn.second))
      {
        found = true;
        s.insert(conn.first);
        break;
      }
    }
    if (!found)
    {
      std::set<size_t> s;
      s.insert(conn.first);
      s.insert(conn.second);
      partition.push_back(s);
    }
  }
  for (auto & p: partition)
  {
    size_t ss = 0;
    for (auto& c : p)
    {
      if (clusterRecordsGlob[c])
      {
        ss += clusterRecordsGlob[c]->indices.size();
      }
    }
    if (
      static_cast<int> (ss) >= min_cluster_size_ &&
      static_cast<int> (ss) <= max_cluster_size_)
    {
        pcl::PointIndices pi;
        pi.header = input_->header;
        pi.indices.resize (ss);

        auto pii = pi.indices.begin();
        
        for (auto& c : p)
        {

          if (clusterRecordsGlob[c])
          {
            pii=std::copy(clusterRecordsGlob[c]->indices.begin(), clusterRecordsGlob[c]->indices.end(), pii);
            clusterRecordsGlob[c].reset();
          }

        }
        clusters.push_back (pi);
    }
    else
      for (auto& c : p)
        clusterRecordsGlob[c].reset();
  }
  //for (auto& c : clusterRecordsGlob)
  for (size_t i=0;i<max_cluster_index;++i)
  {
    auto c = clusterRecordsGlob[i];
    if(c)
    if (
      static_cast<int> (c->indices.size()) >= min_cluster_size_ &&
      static_cast<int> (c->indices.size()) <= max_cluster_size_)
    {
      pcl::PointIndices pi;
      pi.header = input_->header;
      pi.indices.resize (c->indices.size());

      std::copy(c->indices.begin(), c->indices.end(), pi.indices.begin());
      clusters.push_back (pi);
    }
  }


  deinitCompute ();
}





#define PCL_INSTANTIATE_ConditionalEuclideanClustering(T) template class PCL_EXPORTS pcl::ConditionalEuclideanClustering<T>;

#endif  // PCL_SEGMENTATION_IMPL_CONDITIONAL_EUCLIDEAN_CLUSTERING_HPP_


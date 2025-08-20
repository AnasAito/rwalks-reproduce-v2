#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>

namespace hnswlib
{
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template <typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t>
    {
    public:
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

        size_t max_elements_{0};
        mutable std::atomic<size_t> cur_element_count{0}; // current number of elements
        size_t size_data_per_element_{0};
        size_t size_data_attr_per_element_{0};
        size_t size_data_attr_agg_per_element_{0};
        size_t size_links_per_element_{0};
        mutable std::atomic<size_t> num_deleted_{0}; // number of deleted elements
        size_t M_{0};
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t ef_construction_{0};
        size_t ef_{0};
        float hybrid_factor_{0.0f};
        float pron_factor_{0.0f};
        int search_mode{0};
        // float sum_agg_{0.0f};
        // float sum_sq_agg_{0.0f};
        std::vector<float> sum_agg;
        std::vector<float> sum_sq_agg;
        std::vector<float> max_agg;
        std::vector<float> min_agg;

        double mult_{0.0}, revSize_{0.0};
        int maxlevel_{0};

        VisitedListPool *visited_list_pool_{nullptr};

        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_;

        std::mutex global;
        std::vector<std::mutex> link_list_locks_;

        tableint enterpoint_node_{0};

        size_t size_links_level0_{0};
        size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};
        // std::random_device rd;
        // std::default_random_engine rng;

        char *data_level0_memory_{nullptr};
        char *data_attr_memory_{nullptr};
        char *data_attr_agg_memory_{nullptr};
        char **linkLists_{nullptr};
        std::vector<int> element_levels_; // keeps level of each element
        int seedrandom;
        size_t data_size_{0};
        size_t data_attr_size_{0};
        size_t data_attr_agg_size_{0};

        DISTFUNC<dist_t> fstdistfunc_;
        DISTFUNC<int> fstdistattrfunc_;
        DISTFUNC<float> fstdistattraggfunc_;
        void *dist_func_param_{nullptr};
        void *dist_attr_func_param_{nullptr};

        mutable std::mutex label_lookup_lock; // lock for label_lookup_
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};

        bool allow_replace_deleted_ = false; // flag to replace deleted elements (marked as deleted) during insertions

        std::mutex deleted_elements_lock;              // lock for deleted_elements
        std::unordered_set<tableint> deleted_elements; // contains internal ids of deleted elements

        size_t nhops_size_{0};
        size_t cand_valid_ratio_size_{0};

        mutable std::atomic<int> curr_metric{0};
        // metric_Aggregator
        mutable std::vector<std::tuple<int, float, int>> test_metrics;
        mutable std::vector<float> distance_logger;
        HierarchicalNSW(SpaceInterface<dist_t> *s)
        {
        }

        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            const std::string &location,
            bool nmslib = false,
            size_t max_elements = 0,
            bool allow_replace_deleted = false)
            : allow_replace_deleted_(allow_replace_deleted)
        {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(
            SpaceInterface<dist_t> *s,
            size_t max_elements,
            size_t M = 16,
            size_t ef_construction = 200,
            size_t random_seed = 100,
            bool allow_replace_deleted = false)
            : link_list_locks_(max_elements),
              label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
              element_levels_(max_elements),
              allow_replace_deleted_(allow_replace_deleted),
              seedrandom(random_seed)
        {
            max_elements_ = max_elements;
            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            data_attr_size_ = s->get_data_attr_size();
            data_attr_agg_size_ = s->get_data_attr_agg_size();
            fstdistfunc_ = s->get_dist_func();
            fstdistattrfunc_ = s->get_dist_func_attr();
            fstdistattraggfunc_ = s->get_dist_func_attr_agg();
            dist_func_param_ = s->get_dist_func_param();
            dist_attr_func_param_ = s->get_dist_attr_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;

            sum_agg.resize(data_attr_size_, 0.0f);
            sum_sq_agg.resize(data_attr_size_, 0.0f);
            max_agg.resize(data_attr_size_, -50.0f);
            min_agg.resize(data_attr_size_, 50.0f);

            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");
            // --- attr data pre-allocation
            size_data_attr_per_element_ = data_attr_size_;
            data_attr_memory_ = (char *)malloc(max_elements_ * size_data_attr_per_element_);
            if (data_attr_memory_ == nullptr)
                throw std::runtime_error("Not enough memory (Attr data)");

            size_data_attr_agg_per_element_ = data_attr_agg_size_; // dist_attr_func_param_ * sizeof(float);
            data_attr_agg_memory_ = (char *)malloc(max_elements_ * size_data_attr_agg_per_element_);
            if (data_attr_agg_memory_ == nullptr)
                throw std::runtime_error("Not enough memory (Attr data agg)");
            // --- end attr data pre-allocation
            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements_); // each element points to a boid pointer => will evebtually host pointer to memory of its neighbors (link_list)
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;

            // metrics params
            nhops_size_ = sizeof(int);
            cand_valid_ratio_size_ = sizeof(int);
            // size_metric_per_element_ = nhops_size_ + cand_valid_ratio_size_;
        }

        ~HierarchicalNSW()
        {
            free(data_level0_memory_);
            free(data_attr_memory_);
            for (tableint i = 0; i < cur_element_count; i++)
            {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        struct CompareByFirst
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };
        struct CompareByFirstBis
        {
            constexpr bool operator()(std::pair<std::pair<dist_t, bool>, tableint> const &a,
                                      std::pair<std::pair<dist_t, bool>, tableint> const &b) const noexcept
            {
                return a.first.first < b.first.first;
            }
        };

        void setEf(size_t ef)
        {
            ef_ = ef;
        }
        void setHybridFactor(float factor)
        {
            hybrid_factor_ = factor;
        }

        void setPronFactor(float factor)
        {
            pron_factor_ = factor;
        }
        inline std::mutex &getLabelOpMutex(labeltype label) const
        {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
            return label_op_locks_[lock_id];
        }

        inline labeltype getExternalLabel(tableint internal_id) const
        {
            labeltype return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const
        {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const
        {
            return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const
        {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }
        inline char *getDataAttrByInternalId(tableint internal_id) const
        {
            return (data_attr_memory_ + internal_id * size_data_attr_per_element_);
        }
        inline char *getDataAttrAggByInternalId(tableint internal_id) const
        {
            return (data_attr_agg_memory_ + internal_id * size_data_attr_agg_per_element_);
        }

        int getRandomLevel(double reverse_size)
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

        size_t getMaxElements()
        {
            return max_elements_;
        }

        size_t getCurrentElementCount()
        {
            return cur_element_count;
        }

        size_t getDeletedCount()
        {
            return num_deleted_;
        }
        // void mergePriorityQueues(std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &pq1,
        //                          std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &pq2,
        //                          std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &mergedPQ)
        // {
        //     while (!pq1.empty())
        //     {
        //         mergedPQ.push(pq1.top());
        //         pq1.pop();
        //     }

        //     while (!pq2.empty())
        //     {
        //         mergedPQ.push(pq2.top());
        //         pq2.pop();
        //     }
        // }
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer)
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id); // we use emplace to constryct a "pair" without construct and copy
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();                       // pop cand with smallest distance ( max -dist)
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) // early termination
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second; // get current element (best node in our pq st to distance to query)

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                {
                    data = (int *)get_linklist0(curNodeNum);
                }
                else
                {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }
        // int getIndexOfFirstNonNull(const void *data_point_attr, const void *size_of_list_ptr)
        //     const
        // {
        //     size_t size_of_list = *((size_t *)size_of_list_ptr);
        //     const int *list = (const int *)data_point_attr; // Assuming the list contains integers; adjust the type accordingly

        //     for (size_t i = 0; i < size_of_list; ++i)
        //     {
        //         if (list[i] != 0)
        //         {             // Change the condition based on what "non-null" means for your data type
        //             return i; // Return the index of the first non-null value
        //         }
        //     }

        //     return -1; // Return -1 if no non-null vfalue is found
        // }
        int *getAllValidIndices(const void *data_point_attr, const void *size_of_list_ptr)
            const
        {
            size_t size_of_list = *((size_t *)size_of_list_ptr);
            const int *list = (const int *)data_point_attr; // Assuming the list contains integers; adjust the type accordingly

            int *valid_indices = nullptr;
            int num_valid_indices = 0;

            for (size_t i = 0; i < size_of_list; ++i)
            {
                if (list[i] != 0)
                {
                    int *temp = (int *)realloc(valid_indices, (num_valid_indices + 1) * sizeof(int));
                    if (temp == nullptr)
                    {
                        free(valid_indices);
                        return nullptr; // Allocation failure
                    }
                    valid_indices = temp;
                    valid_indices[num_valid_indices++] = i;
                }
            }

            return valid_indices; // Return the dynamically allocated array of valid indices
        }
        int getAllValidIndicesCount(const void *data_point_attr, const void *size_of_list_ptr)
            const
        {
            size_t size_of_list = *((size_t *)size_of_list_ptr);
            const int *list = (const int *)data_point_attr; // Assuming the list contains integers; adjust the type accordingly

            int num_valid_indices = 0;

            for (size_t i = 0; i < size_of_list; ++i)
            {
                if (list[i] != 0)
                    num_valid_indices++;
            }

            return num_valid_indices; // Return the dynamically allocated array of valid indices
        }

        float calculateMean(const std::vector<float> &data)
            const
        {
            float sum = 0.0f;
            for (const auto &element : data)
            {
                sum += element;
            }
            return sum / data.size();
        }

        float calculateStandardDeviation(const std::vector<float> &data)
            const
        {
            if (data.size() < 2)
            {
                // Standard deviation is undefined for less than two elements
                return 0.0f;
            }

            float mean = calculateMean(data);

            // Calculate the sum of squared differences from the mean
            float sumSquaredDifferences = 0.0f;
            for (const auto &element : data)
            {
                float difference = element - mean;
                sumSquaredDifferences += difference * difference;
            }

            // Calculate the mean of squared differences
            float meanSquaredDifferences = sumSquaredDifferences / data.size();

            // Take the square root to get the standard deviation
            float standardDeviation = std::sqrt(meanSquaredDifferences);

            return standardDeviation;
        }

        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr, const void *data_point_attr = nullptr, const bool collect_metrics = false) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::string search_method = "dpq";

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                top_valid_only_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            std::pair<dist_t, tableint> topElement;

            char *currObj1Attr;
            int distAttr;
            bool isCandAttrAllowed;

            dist_t lowerBound;

            // int query_Attr_valid_index = getIndexOfFirstNonNull(data_point_attr, dist_attr_func_param_);
            int valid_indices_count = getAllValidIndicesCount(data_point_attr, dist_attr_func_param_);
            int *valid_indices = getAllValidIndices(data_point_attr, dist_attr_func_param_);
            // std::cout << "query_Attr_valid_index: " << query_Attr_valid_index << std::endl;

            if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))
            {

                char *currObj1 = (getDataByInternalId(ep_id));

                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                char *currObj1AttrAgg = (getDataAttrAggByInternalId(ep_id));
                if (hybrid_factor_ > 0)
                {
                    dist_t distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
                    dist += hybrid_factor_ * distAttrAgg;
                }

                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            int nhops = 0;
            int valid_hops = 0;
            int dist_calculations = 0;
            dist_t distAttrAgg;
            std::vector<float> dist_per_q;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions)))
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);

                if (collect_metrics)
                {
                    nhops++;
                    char *currObj1Attr = (getDataAttrByInternalId(current_node_id));
                    int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    if (distAttr == 0)
                    {
                        valid_hops++;
                    }
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif
                std::vector<float> nei_dists(size, 0.0f);
                int visited_count = 0;
                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif

                    bool pron_condition;
                    if (hybrid_factor_ > 0)
                    {
                        char *currObj1AttrAgg = (getDataAttrAggByInternalId(candidate_id));
                        distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
                        float *pVect2 = (float *)currObj1AttrAgg;
                        pron_condition = (pVect2[valid_indices[0]] >= pron_factor_);
                        if (valid_indices_count > 1)
                        {
                            pron_condition = pron_condition && (pVect2[valid_indices[1]] >= pron_factor_);
                        }
                    }
                    else
                    {
                        distAttrAgg = 0;
                        pron_condition = true;
                    }

                    if (!(visited_array[candidate_id] == visited_array_tag) && pron_condition)
                    {
                        // if (d_a < 1 and d_b < 1)
                        // {
                        //     std::cout << d_a << " / " << d_b << std::endl;
                        // }
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        dist += hybrid_factor_ * distAttrAgg;
                        if (collect_metrics)
                        {
                            dist_calculations++;
                            // distance_logger.push_back(fstdistfunc_(data_point, currObj1, dist_func_param_));
                            nei_dists.push_back(fstdistfunc_(data_point, currObj1, dist_func_param_));
                            visited_count++;
                        }

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                            {
                                top_candidates.emplace(dist, candidate_id);
                            }

                            if (top_candidates.size() > ef)
                            {

                                if (search_method == "dpq")
                                { // add poped element to top_valid_only_candidates if isCandAttrAllowed
                                    // std::cout << "top_cand overflow ... " << nhops << std::endl;
                                    topElement = top_candidates.top();
                                    currObj1Attr = (getDataAttrByInternalId(topElement.second));
                                    distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                                    isCandAttrAllowed = (distAttr == 0);
                                    if (isCandAttrAllowed)
                                    {
                                        // std::cout << "updating dpq ..." << std::endl;
                                        top_valid_only_candidates.emplace(topElement);
                                    }
                                }

                                top_candidates.pop();
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
                if (collect_metrics)
                {
                    // float stdDeviation = calculateStandardDeviation(nei_dists);
                    // float mean_dist = calculateMean(nei_dists);
                    // distance_logger.push_back(mean_dist);
                    float visited_ratio = (float)visited_count / (float)size;
                    distance_logger.push_back(visited_ratio);
                    // distance_logger.push_back((float)nhops);
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            if (collect_metrics)
            {
                float satisfied_ratio = (float)valid_hops / (float)nhops;
                std::cout << " dist_calculations = " << dist_calculations << std::endl;
                std::cout << "---------------------------" << std::endl;
                test_metrics.push_back(std::make_tuple(dist_calculations, satisfied_ratio, nhops));
                curr_metric++;
            }
            // TODO - merge pqs
            // !  check python implmt (cant reproduce recall results  - maybe adding elements to top_cand should be changed)

            // std::cout << "top_candidates.size() = " << top_candidates.size() << " top_valid_only_candidates.size() = " << top_valid_only_candidates.size();

            // merge pqs
            // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            //     top_merged_candidates;
            // return top_candidates;

            // // s_n_f
            // while (!top_candidates.empty())
            // {
            //     topElement = top_candidates.top();
            //     currObj1Attr = (getDataAttrByInternalId(topElement.second));
            //     // distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
            //     distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
            //     // distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
            //     isCandAttrAllowed = (distAttr == 0);
            //     if (isCandAttrAllowed)
            //         top_valid_only_candidates.emplace(topElement);
            //     else
            //         top_valid_only_candidates.emplace(300000.0f + topElement.first, topElement.second);
            //     top_candidates.pop();
            // }
            // return top_valid_only_candidates;
            if (search_method == "dpq")
            {
                // double_pq
                while (!top_candidates.empty())
                {
                    topElement = top_candidates.top();
                    currObj1Attr = (getDataAttrByInternalId(topElement.second));
                    // distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                    // distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                    distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    isCandAttrAllowed = (distAttr == 0);
                    if (isCandAttrAllowed)
                        top_valid_only_candidates.emplace(topElement);
                    top_candidates.pop();
                }
                return top_valid_only_candidates;
            }
            if (search_method == "snf")
            {
                // s_n_f
                while (!top_candidates.empty())
                {
                    topElement = top_candidates.top();
                    currObj1Attr = (getDataAttrByInternalId(topElement.second));
                    // distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                    distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    // distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                    isCandAttrAllowed = (distAttr == 0);
                    if (isCandAttrAllowed)
                        top_valid_only_candidates.emplace(topElement);
                    else
                        top_valid_only_candidates.emplace(300000.0f + topElement.first, topElement.second);
                    top_candidates.pop();
                }
                return top_valid_only_candidates;
            }
        }
        template <bool has_deletions>
        std::priority_queue<std::pair<std::pair<dist_t, bool>, tableint>, std::vector<std::pair<std::pair<dist_t, bool>, tableint>>, CompareByFirstBis>
        searchBaseLayerSTAttrOnly(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr, const void *data_point_attr = nullptr, const bool collect_metrics = false) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::string search_method = "dpq";

            std::priority_queue<std::pair<std::pair<dist_t, bool>, tableint>, std::vector<std::pair<std::pair<dist_t, bool>, tableint>>, CompareByFirstBis>
                top_valid_only_candidates;
            std::priority_queue<std::pair<std::pair<dist_t, bool>, tableint>, std::vector<std::pair<std::pair<dist_t, bool>, tableint>>, CompareByFirstBis> top_candidates;
            std::priority_queue<std::pair<std::pair<dist_t, bool>, tableint>, std::vector<std::pair<std::pair<dist_t, bool>, tableint>>, CompareByFirstBis> candidate_set;
            std::pair<std::pair<dist_t, bool>, tableint> topElement;

            char *currObj1Attr;
            int distAttr;
            dist_t distAttrAgg;
            bool isCandAttrAllowed;

            dist_t lowerBound;
            int valid_indices_count = getAllValidIndicesCount(data_point_attr, dist_attr_func_param_);
            int *valid_indices = getAllValidIndices(data_point_attr, dist_attr_func_param_);
            // std::cout << "valid_indices_count: " << valid_indices_count << std::endl;
            // std::cout << "valid_indices: " << valid_indices << std::endl;
            char *currObj1 = (getDataByInternalId(ep_id));

            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

            char *currObj1AttrAgg = (getDataAttrAggByInternalId(ep_id));
            distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
            if (hybrid_factor_ > 0)
            {
                dist -= hybrid_factor_ * (distAttrAgg > 5 ? distAttrAgg - 5 : distAttrAgg);
            }

            lowerBound = dist;
            top_candidates.emplace(std::make_pair(dist, distAttrAgg >= valid_indices_count * 5), ep_id);
            candidate_set.emplace(std::make_pair(-dist, true), ep_id);

            visited_array[ep_id] = visited_array_tag;

            int nhops = 0;
            int point_attr_eval = 0;
            int valid_hops = 0;
            int dist_calculations = 0;

            // std::vector<float> dist_per_q;

            while (!candidate_set.empty())
            {

                std::pair<std::pair<dist_t, bool>, tableint> current_node_pair = candidate_set.top();

                // if ((-current_node_pair.first.first) > lowerBound &&
                //     (top_candidates.size() == ef || (!isIdAllowed && !has_deletions)))
                if ((-current_node_pair.first.first) > lowerBound && top_candidates.size() == ef)
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                if (collect_metrics)
                {
                    nhops++;
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif
                // std::vector<float> nei_dists(size, 0.0f);
                int visited_count = 0;
                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif

                    bool is_not_visited = !(visited_array[candidate_id] == visited_array_tag);
                    bool node_to_probe = is_not_visited;
                    // if (hybrid_factor_ > 0)
                    // {

                    // if (is_not_visited)
                    // {
                    // distAttrAgg = fstdistattraggfunc_(query_label, mapRefs.find(candidate_id)->second);

                    if (pron_factor_ != -1 && is_not_visited)
                    {
                        char *currObj1AttrAgg = (getDataAttrAggByInternalId(candidate_id));
                        distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
                        node_to_probe = node_to_probe && (distAttrAgg / valid_indices_count >= pron_factor_);
                        visited_array[candidate_id] = visited_array_tag;

                        if (collect_metrics)
                        {
                            point_attr_eval++;
                        }
                    }

                    if (node_to_probe)
                    {
                        // visited_array[candidate_id] = visited_array_tag;
                        char *currObj1 = (getDataByInternalId(candidate_id));

                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        if (hybrid_factor_ > 0)
                            dist -= hybrid_factor_ * (distAttrAgg > 5 ? distAttrAgg - 5 : distAttrAgg);

                        // std::cout << "dist = " << dist << std::endl;
                        // std::cout << "distAttrAgg = " << distAttrAgg << " normed = " << (distAttrAgg > 5 ? distAttrAgg - 5 : distAttrAgg) << std::endl;
                        if (collect_metrics)
                        {
                            dist_calculations++;
                            // nei_dists.push_back(fstdistfunc_(data_point, currObj1, dist_func_param_));
                            // visited_count++;
                        }

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(std::make_pair(-dist, true), candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            // if ((!has_deletions || !isMarkedDeleted(candidate_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                            // {
                            // std::cout <<"dist = "<< distAttrAgg << std::endl;
                            top_candidates.emplace(std::make_pair(dist, distAttrAgg >= 5 * valid_indices_count), candidate_id);
                            // }

                            if (top_candidates.size() > ef)
                            {

                                if (search_method == "dpq")
                                {
                                    topElement = top_candidates.top();
                                    // distAttr = fstdistattrfunc_((void *)valid_indices, (void *)&valid_indices_count, mapRefs.find(topElement.second)->second);
                                    // isCandAttrAllowed = (distAttr == 0);
                                    isCandAttrAllowed = topElement.first.second;
                                    if (isCandAttrAllowed)
                                        top_valid_only_candidates.emplace(topElement);
                                }

                                top_candidates.pop();
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first.first;
                        }
                    }
                }
                if (collect_metrics)
                {

                    // float visited_ratio = (float)visited_count / (float)size;
                    distance_logger.push_back(dist_calculations);
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            if (collect_metrics)
            {
                test_metrics.push_back(std::make_tuple(dist_calculations, point_attr_eval, nhops));

                curr_metric++;
            }

            if (search_method == "dpq")
            {
                // double_pq
                while (!top_candidates.empty())
                {
                    topElement = top_candidates.top();
                    // distAttr = fstdistattrfunc_((void *)valid_indices, (void *)&valid_indices_count, mapRefs.find(topElement.second)->second);
                    // isCandAttrAllowed = (distAttr == 0);
                    isCandAttrAllowed = topElement.first.second;
                    if (isCandAttrAllowed)
                        top_valid_only_candidates.emplace(topElement);
                    top_candidates.pop();
                }

                return top_valid_only_candidates;
            }
            if (search_method == "snf")
            {
                // s_n_f : search then filter
                while (!top_candidates.empty())
                {
                    topElement = top_candidates.top();
                    // distAttr = fstdistattrfunc_((void *)valid_indices, (void *)&valid_indices_count, mapRefs.find(topElement.second)->second);
                    // isCandAttrAllowed = (distAttr == 0);
                    isCandAttrAllowed = topElement.first.second;
                    if (isCandAttrAllowed)
                        top_valid_only_candidates.emplace(topElement);
                    else
                        top_valid_only_candidates.emplace(std::make_pair(300000.0f + topElement.first.first, false), topElement.second);
                    top_candidates.pop();
                }
                return top_valid_only_candidates;
            }
        }
        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerSTFpq(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr, const void *data_point_attr = nullptr, const bool collect_metrics = false) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            float dist_comp = 0;

            // int query_Attr_valid_index = getIndexOfFirstNonNull(data_point_attr, dist_attr_func_param_);
            int valid_indices_count = getAllValidIndicesCount(data_point_attr, dist_attr_func_param_);
            int *valid_indices = getAllValidIndices(data_point_attr, dist_attr_func_param_);
            if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            int nhops = 0;
            int valid_hops = 0;
            int attr_eval_points = 0;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions)))
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                // if (collect_metrics)
                // {
                //     char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                //     int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                //     if (distAttr != 1)
                //     {
                //         valid_hops++;
                //     }
                // }
                if (collect_metrics)
                {
                    nhops++;
                    // metric_distance_computations += size;
                    char *currObj1Attr = (getDataAttrByInternalId(current_node_id));
                    // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                    // int distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                    int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    // if (distAttr == 0)
                    // {
                    //     valid_hops++;
                    // }
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (collect_metrics)
                        {
                            attr_eval_points++;
                            dist_comp = dist_comp + 1.0f;
                        }

                        // std::cout << "dist_vect = " << dist << "/ dist_attr =" << distAttr << std::endl;
                        char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                        // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                        // int distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                        int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                        bool isCandAttrAllowed = (distAttr == 0);
                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))) && isCandAttrAllowed)
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            if (collect_metrics)
            {
                float satisfied_ratio = (float)valid_hops / (float)nhops;
                // std::cout << "nhops = " << nhops << " valid_hops = " << valid_hops << " satisfied_ratio = " << satisfied_ratio << std::endl;
                // test_metrics.push_back(std::make_pair(nhops, (float)dist_comp));
                test_metrics.push_back(std::make_tuple(dist_comp, attr_eval_points, nhops));
                distance_logger.push_back(dist_comp);
                curr_metric++;
            }
            return top_candidates;
        }
        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerSTSnfRobust(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr, const void *data_point_attr = nullptr, const bool collect_metrics = false) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_valid_only;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            float dist_comp = 0;

            // int query_Attr_valid_index = getIndexOfFirstNonNull(data_point_attr, dist_attr_func_param_);
            int valid_indices_count = getAllValidIndicesCount(data_point_attr, dist_attr_func_param_);
            int *valid_indices = getAllValidIndices(data_point_attr, dist_attr_func_param_);
            if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            int nhops = 0;
            int valid_hops = 0;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions)))
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                // if (collect_metrics)
                // {
                //     char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                //     int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                //     if (distAttr != 1)
                //     {
                //         valid_hops++;
                //     }
                // }
                if (collect_metrics)
                {
                    nhops++;
                    // metric_distance_computations += size;
                    char *currObj1Attr = (getDataAttrByInternalId(current_node_id));
                    // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                    // int distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                    int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    if (distAttr == 0)
                    {
                        valid_hops++;
                    }
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (collect_metrics)
                        {
                            // char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                            // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                            // if (distAttr != 1)
                            // {
                            //     valid_hops++;
                            // }
                            dist_comp = dist_comp + 1.0f;
                        }

                        // std::cout << "dist_vect = " << dist << "/ dist_attr =" << distAttr << std::endl;

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            if (collect_metrics)
            {
                float satisfied_ratio = (float)valid_hops / (float)nhops;
                // std::cout << "nhops = " << nhops << " valid_hops = " << valid_hops << " satisfied_ratio = " << satisfied_ratio << std::endl;
                // test_metrics.push_back(std::make_pair(nhops, (float)dist_comp));
                test_metrics.push_back(std::make_tuple(dist_comp, satisfied_ratio, nhops));
                distance_logger.push_back(dist_comp);
                curr_metric++;
            }

            // s_n_f : search then filter

            int k_prime = 100;
            while (top_candidates.size() > k_prime)
            {
                top_candidates.pop();
            }
            while (!top_candidates.empty())
            {
                auto topElement = top_candidates.top();
                char *currObj1Attr = (getDataAttrByInternalId(topElement.second));
                int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                bool isCandAttrAllowed = (distAttr == 0);
                if (isCandAttrAllowed)
                    top_candidates_valid_only.emplace(topElement);
                top_candidates.pop();
            }
            return top_candidates_valid_only;
            // return top_candidates;
        }
        template <bool has_deletions>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerSTSnf(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor *isIdAllowed = nullptr, const void *data_point_attr = nullptr, const bool collect_metrics = false) const
        {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_valid_only;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            float dist_comp = 0;

            // int query_Attr_valid_index = getIndexOfFirstNonNull(data_point_attr, dist_attr_func_param_);
            int valid_indices_count = getAllValidIndicesCount(data_point_attr, dist_attr_func_param_);
            int *valid_indices = getAllValidIndices(data_point_attr, dist_attr_func_param_);
            if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))
            {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            int nhops = 0;
            int valid_hops = 0;

            while (!candidate_set.empty())
            {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions)))
                {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *)get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint *)data);
                //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                // if (collect_metrics)
                // {
                //     char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                //     int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                //     if (distAttr != 1)
                //     {
                //         valid_hops++;
                //     }
                // }
                if (collect_metrics)
                {
                    nhops++;
                    // metric_distance_computations += size;
                    char *currObj1Attr = (getDataAttrByInternalId(current_node_id));
                    // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                    // int distAttr = fstdistattrfunc_((void *)&query_Attr_valid_index, currObj1Attr, dist_attr_func_param_);
                    int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
                    if (distAttr == 0)
                    {
                        valid_hops++;
                    }
                }

#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++)
                {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0); ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag))
                    {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (collect_metrics)
                        {
                            // char *currObj1Attr = (getDataAttrByInternalId(candidate_id));
                            // int distAttr = fstdistattrfunc_(data_point_attr, currObj1Attr, dist_attr_func_param_);
                            // if (distAttr != 1)
                            // {
                            //     valid_hops++;
                            // }
                            dist_comp = dist_comp + 1.0f;
                        }

                        // std::cout << "dist_vect = " << dist << "/ dist_attr =" << distAttr << std::endl;

                        if (top_candidates.size() < ef || lowerBound > dist)
                        {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                             offsetLevel0_, ///////////
                                         _MM_HINT_T0);      ////////////////////////
#endif

                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            if (collect_metrics)
            {
                float satisfied_ratio = (float)valid_hops / (float)nhops;
                // std::cout << "nhops = " << nhops << " valid_hops = " << valid_hops << " satisfied_ratio = " << satisfied_ratio << std::endl;
                // test_metrics.push_back(std::make_pair(nhops, (float)dist_comp));
                test_metrics.push_back(std::make_tuple(dist_comp, satisfied_ratio, nhops));
                distance_logger.push_back(dist_comp);
                curr_metric++;
            }

            // s_n_f : search then filter

            // int k_prime = 100;
            // while (top_candidates.size() > k_prime)
            // {
            //     top_candidates.pop();
            // }
            // while (!top_candidates.empty())
            // {
            //     auto topElement = top_candidates.top();
            //     char *currObj1Attr = (getDataAttrByInternalId(topElement.second));
            //     int distAttr = fstdistattrfunc_((void *)valid_indices, currObj1Attr, (void *)&valid_indices_count);
            //     bool isCandAttrAllowed = (distAttr == 0);
            //     if (isCandAttrAllowed)
            //         top_candidates_valid_only.emplace(topElement);
            //     top_candidates.pop();
            // }
            // return top_candidates_valid_only;
            return top_candidates;
        }
        std::vector<tableint>
        getGraphWalk(tableint start_node, int walk_length) const
        {
            std::vector<tableint> walk;
            int step = 0;
            tableint current_node_id = start_node;
            std::unordered_set<tableint> visited_nodes;

            while (step < walk_length)
            {
                // pull pointer to node neighbors
                int *neighbors = (int *)get_linklist0(current_node_id);

                // choose a neighbor idx excluding visited ones
                size_t size = getListCount((linklistsizeint *)neighbors);
                std::vector<size_t> unvisited_indices;

                for (size_t i = 0; i < size; ++i)
                {
                    tableint neighbor_id = *(neighbors + i);
                    if (visited_nodes.find(neighbor_id) == visited_nodes.end())
                    {
                        unvisited_indices.push_back(i);
                    }
                }

                if (unvisited_indices.empty())
                {
                    // All neighbors have been visited, reset visited set
                    visited_nodes.clear();
                    unvisited_indices.resize(size);
                    std::iota(unvisited_indices.begin(), unvisited_indices.end(), 0);
                }

                size_t random_idx = unvisited_indices[std::rand() % unvisited_indices.size()];

                // push to walk
                current_node_id = *(neighbors + random_idx);

                walk.push_back(current_node_id);
                visited_nodes.insert(current_node_id);
                step++;
                // std::cout << "current_node_id = " << current_node_id << "/ random idx " << random_idx << "/ size " << size << std::endl;
            }
            // std::cout << "walk.size() = " << walk.size() << std::endl;
            return walk;
        }
        std::vector<tableint> getGraphWalk2(tableint start_node, int walk_length) const
        {
            std::vector<tableint> walk;
            walk.reserve(walk_length);

            tableint current_node_id = start_node;
            std::unordered_set<tableint> visited_nodes;
            visited_nodes.reserve(walk_length);
            std::random_device rd;
            std::default_random_engine rng(rd());
            for (int step = 0; step < walk_length; step++)
            {

                auto neighbors = get_linklist0(current_node_id);
                auto size = getListCount((linklistsizeint *)neighbors);
                // std::random_device rd;
                // std::default_random_engine rng(rd());
                std::vector<tableint> shuffled_neighbors(neighbors, neighbors + size);
                std::shuffle(shuffled_neighbors.begin(), shuffled_neighbors.end(), rng);

                tableint next_node_id = current_node_id;

                for (tableint neighbor_id : shuffled_neighbors)
                {
                    if (visited_nodes.find(neighbor_id) == visited_nodes.end())
                    {
                        next_node_id = neighbor_id;
                        break;
                    }
                }
                if (next_node_id == current_node_id)
                {
                    visited_nodes.clear();
                    next_node_id = shuffled_neighbors[std::rand() % size];
                }

                walk.push_back(next_node_id);
                visited_nodes.insert(next_node_id);
                current_node_id = next_node_id;
            }
            return walk;
        }
        std::vector<tableint> getGraphWalk3(tableint start_node, int walk_length) const
        {
            std::vector<tableint> walk;
            walk.reserve(walk_length);

            tableint current_node_id = start_node;
            std::unordered_set<tableint> visited_nodes;
            visited_nodes.reserve(walk_length);

            std::random_device rd;
            std::default_random_engine rng(rd());

            for (int step = 0; step < walk_length; ++step)
            {
                auto neighbors = get_linklist0(current_node_id);
                auto size = getListCount((linklistsizeint *)neighbors);

                std::uniform_int_distribution<int> dist(0, size - 1);

                tableint next_node_id = current_node_id;

                for (int i = 0; i < size; ++i)
                {
                    int random_index = dist(rng);
                    tableint neighbor_id = neighbors[random_index];

                    if (visited_nodes.find(neighbor_id) == visited_nodes.end())
                    {
                        next_node_id = neighbor_id;
                        break;
                    }
                }

                if (next_node_id == current_node_id)
                {
                    visited_nodes.clear();
                    next_node_id = neighbors[dist(rng)];
                }

                walk.push_back(next_node_id);
                visited_nodes.insert(next_node_id);
                current_node_id = next_node_id;
            }

            return walk;
        }
        // Function to calculate the decay sum
        float *calculateDecaySum(const std::vector<tableint> &walk, float epsilon)
        {

            size_t dim_attr = *((size_t *)dist_attr_func_param_);
            float *result = new float[dim_attr](); // Initialize array to zeros

            for (int k = 0; k < walk.size(); ++k)
            {
                char *node_attr_ptr = (getDataAttrByInternalId(walk[k]));
                int *node_attr = (int *)node_attr_ptr;
                for (int i = 0; i < dim_attr; ++i)
                {
                    if (*(node_attr + i) == 1)
                    {
                        result[i] += std::pow(epsilon, k);
                        // if (dim_attr >= 1038 && result[i] > 0)
                        // {
                        //     std::cout << " vec increased = " << result[i] << std::endl;
                        // }
                    }
                }
            }
            for (int i = 0; i < dim_attr; ++i)
            {

                result[i] /= walk.size();
            }

            return result;
        }

        void displayArray(float *vec, int dim)
        {
            for (int i = 0; i < dim; ++i)
            {

                std::cout << vec[i] << " ";
            }

            std::cout << std::endl;
        }

        void getAttrAggregate(tableint node_id, int walk_length, int walk_count, float decay_factor)
        {
            int trackInterval = 100000;
            if ((node_id + 1) % trackInterval == 0)
            {
                auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                auto tm = std::localtime(&now);

                std::cout << "(RW) Progress: " << (node_id + 1) << " iterations completed. Current time: "
                          << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << std::endl;
            }
            // int walk_count = 10;
            int dim_attr = *((int *)dist_attr_func_param_);
            float *vec_attr_agg = new float[dim_attr]();

            char *node_attr_ptr = (getDataAttrByInternalId(node_id));
            int *node_attr = (int *)node_attr_ptr;

            for (int i = 0; i < walk_count; ++i)
            {
                std::vector<tableint> walk = getGraphWalk2(node_id, walk_length);
                // if (node_id == 0)
                // {
                //     std::vector<tableint> walk2 = getGraphWalk2(node_id, walk_length);
                //     std::cout << "-------------- walkv1" << std::endl;
                //     for (int i = 0; i < walk.size(); ++i)
                //     {
                //         std::cout << walk[i] << " ";
                //     }
                //     std::cout << std::endl;
                //     std::cout << "-------------- walkv2" << std::endl;
                //     for (int i = 0; i < walk2.size(); ++i)
                //     {
                //         std::cout << walk2[i] << " ";
                //     }
                //     std::cout << std::endl;
                // }
                float *vec_attr_agg_per_walk = calculateDecaySum(walk, decay_factor);

                // update vec_attr_agg (sum of vectors comming from  walks)
                for (int j = 0; j < dim_attr; ++j)
                {
                    vec_attr_agg[j] += vec_attr_agg_per_walk[j];
                }

                delete[] vec_attr_agg_per_walk;
            }
            for (int j = 0; j < dim_attr; ++j)
            {
                int vec_attr_val = *(node_attr + j);
                // vec_attr_agg[j] = vec_attr_val;

                if (vec_attr_val == 0)
                {
                    vec_attr_agg[j] = vec_attr_agg[j] / walk_count;
                }

                else
                {
                    vec_attr_agg[j] = 1.0f;
                    // (vec_attr_agg[j] / walk_count) + 0.08f;
                }
            }
            if (node_id == 0)
            {
                std::cout << " vec_attr_agg ---------------------------------" << std::endl;

                displayArray(vec_attr_agg, dim_attr);
                std::cout << "---------------------------------" << std::endl;
            }

            // write to memory
            memcpy(data_attr_agg_memory_ + node_id * size_data_attr_agg_per_element_, vec_attr_agg, size_data_attr_agg_per_element_);

            // std::cout << "final vec_attr_agg:---------------------------------" << std::endl;
            // displayArray(vec_attr_agg, dim_attr);
            // std::cout << "---------------------------------" << std::endl;
            delete[] vec_attr_agg;

            // // check vector in memory
            // char *agg_vec_ptr = getDataAttrAggByInternalId(node_id);
            // // cast to float
            // float *_arr = (float *)agg_vec_ptr;
            // for (int i = 0; i < dim_attr; ++i)
            // {
            //     std::cout << _arr[i] << " ";
            // }
            // std::cout << std::endl;
        }
        float getBinCode(float x, float min, float max, int numBins)
        {
            // Calculate bin width
            double binWidth = (max - min) / numBins;

            // Determine the bin index for the given value
            int binIndex = static_cast<int>((x - min) / binWidth);

            // Ensure the bin index is within bounds
            binIndex = std::max(0, std::min(binIndex, numBins - 1));

            // Return the bin code (bin index)

            return min + binWidth * binIndex;
        }

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            const size_t M)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0)
            {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list)
                {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list)
            {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        linklistsizeint *get_linklist0(tableint internal_id) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const
        {
            return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }

        linklistsizeint *get_linklist(tableint internal_id, int level) const
        {
            return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        }

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const
        {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        }

        tableint mutuallyConnectNewElement(
            const void *data_point,
            tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
            int level,
            bool isUpdate)
        {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                // lock only during the update
                // because during the addition the lock for cur_c is already acquired
                std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
                if (isUpdate)
                {
                    lock.lock();
                }
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *)(ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate)
                {
                    for (size_t j = 0; j < sz_link_list_other; j++)
                    {
                        if (data[j] == cur_c)
                        {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                             dist_func_param_),
                                data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        void resizeIndex(size_t new_max_elements)
        {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location)
        {

            size_t dotBinPos = location.find(".bin");
            std::string attr_location = location.substr(0, dotBinPos) + "_attr.bin";

            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, size_data_attr_per_element_);
            writeBinaryPOD(output, size_data_attr_agg_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++)
            {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();

            std::ofstream attr_output(attr_location, std::ios::binary);
            attr_output.write(data_attr_memory_, cur_element_count * size_data_attr_per_element_);
            attr_output.write(data_attr_agg_memory_, cur_element_count * size_data_attr_agg_per_element_);
            attr_output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0)
        {

            size_t dotBinPos = location.find(".bin");
            std::string attr_location = location.substr(0, dotBinPos) + "_attr.bin";

            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, size_data_attr_per_element_);
            readBinaryPOD(input, size_data_attr_agg_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            // data_size_ = s->get_data_size();
            // data_attr_size_ = s->get_data_attr_size();
            // fstdistfunc_ = s->get_dist_func();
            // dist_func_param_ = s->get_dist_func_param();

            data_size_ = s->get_data_size();
            data_attr_size_ = s->get_data_attr_size();
            fstdistfunc_ = s->get_dist_func();
            fstdistattrfunc_ = s->get_dist_func_attr();
            fstdistattraggfunc_ = s->get_dist_func_attr_agg();
            dist_func_param_ = s->get_dist_func_param();
            dist_attr_func_param_ = s->get_dist_attr_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++)
            {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                }
                else
                {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++)
            {
                if (isMarkedDeleted(i))
                {
                    num_deleted_ += 1;
                    if (allow_replace_deleted_)
                        deleted_elements.insert(i);
                }
            }

            input.close();

            std::ifstream inputbis(attr_location, std::ios::binary);
            if (!inputbis.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            inputbis.seekg(0, inputbis.end);
            total_filesize = inputbis.tellg();
            inputbis.seekg(0, inputbis.beg);

            auto pos_bis = inputbis.tellg();

            // /// Optional - check if index is ok:
            // inputbis.seekg(cur_element_count * (size_data_attr_per_element_ + size_data_attr_agg_per_element_), inputbis.cur);
            // for (size_t i = 0; i < cur_element_count; i++)
            // {
            //     if (inputbis.tellg() < 0 || inputbis.tellg() >= total_filesize)
            //     {
            //         throw std::runtime_error("Index-attr seems to be corrupted or unsupported");
            //     }
            // }

            // // throw exception if it either corrupted or old index
            // if (inputbis.tellg() != total_filesize)
            //     throw std::runtime_error("Index-attr seems to be corrupted or unsupported (2)");

            // inputbis.clear();
            // /// Optional check end

            inputbis.seekg(pos_bis, inputbis.beg);

            data_attr_memory_ = (char *)malloc(max_elements * size_data_attr_per_element_);
            if (data_attr_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate attr");
            inputbis.read(data_attr_memory_, cur_element_count * size_data_attr_per_element_);

            data_attr_agg_memory_ = (char *)malloc(max_elements * size_data_attr_agg_per_element_);
            if (data_attr_agg_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate attr_agg");
            inputbis.read(data_attr_agg_memory_, cur_element_count * size_data_attr_agg_per_element_);
            inputbis.close();
            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (int i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }
        // template <typename data_t>
        std::vector<int> getDataAttrByLabel(labeltype label) const
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataAttrByInternalId(internalId);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<int> data;
            int *data_ptr = (int *)data_ptrv;
            for (int i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }
        std::vector<float> getDataAttrAggByLabel(labeltype label) const
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second))
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataAttrAggByInternalId(internalId);
            size_t dim = *((size_t *)dist_attr_func_param_);
            std::vector<float> data;
            float *data_ptr = (float *)data_ptrv;
            for (int i = 0; i < dim; i++)
            {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }
        /*
         * Marks an element with the given label deleted, does NOT really change the current graph.
         */
        void markDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            markDeletedInternal(internalId);
        }

        /*
         * Uses the last 16 bits of the memory for the linked list size to store the mark,
         * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
         */
        void markDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /*
         * Removes the deleted mark of the node, does NOT really change the current graph.
         *
         * Note: the method is not safe to use when replacement of deleted elements is enabled,
         *  because elements marked as deleted can be completely removed by addPoint
         */
        void unmarkDelete(labeltype label)
        {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end())
            {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }

        /*
         * Remove the deleted mark of the node.
         */
        void unmarkDeletedInternal(tableint internalId)
        {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (allow_replace_deleted_)
                {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /*
         * Checks the first 16 bits of the memory to see if the element is marked deleted.
         */
        bool isMarkedDeleted(tableint internalId) const
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint *ptr) const
        {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint *ptr, unsigned short int size) const
        {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        /*
         * Adds point. Updates the point if it is already in the index.
         * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
         */
        void addPoint(const void *data_point, labeltype label, bool replace_deleted = false, const void *datapoint_attr = nullptr)
        {
            if ((allow_replace_deleted_ == false) && (replace_deleted == true))
            {
                throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
            }

            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            if (!replace_deleted)
            {
                addPoint(data_point, label, -1, datapoint_attr);
                return;
            }
            // check if there is vacant place
            tableint internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
            bool is_vacant_place = !deleted_elements.empty();
            if (is_vacant_place)
            {
                internal_id_replaced = *deleted_elements.begin();
                deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock();

            // if there is no vacant place then add or update point
            // else add point to vacant place
            if (!is_vacant_place)
            {
                addPoint(data_point, label, -1, datapoint_attr);
            }
            else
            {
                // we assume that there are no concurrent operations on deleted element
                labeltype label_replaced = getExternalLabel(internal_id_replaced);
                setExternalLabel(internal_id_replaced, label);

                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                label_lookup_.erase(label_replaced);
                label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                unmarkDeletedInternal(internal_id_replaced);
                updatePoint(data_point, internal_id_replaced, 1.0);
            }
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability)
        {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++)
            {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop : listOneHop)
                {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop : listTwoHop)
                    {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto &&neigh : sNeigh)
                {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto &&cand : sCand)
                    {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep)
                        {
                            candidates.emplace(distance, cand);
                        }
                        else
                        {
                            if (distance < candidates.top().first)
                            {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *)(ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++)
                        {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        }

        void repairConnectionsForUpdate(
            const void *dataPoint,
            tableint entryPointInternalId,
            tableint dataPointInternalId,
            int dataPointLevel,
            int maxLevel)
        {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel)
            {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++)
                        {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist)
                            {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--)
            {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0)
                {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0)
                {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted)
                    {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level)
        {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        }

        tableint addPoint(const void *data_point, labeltype label, int level, const void *datapoint_attr = nullptr)
        {
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end())
                {
                    tableint existingInternalId = search->second;
                    if (allow_replace_deleted_)
                    {
                        if (isMarkedDeleted(existingInternalId))
                        {
                            throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                        }
                    }
                    lock_table.unlock();

                    if (isMarkedDeleted(existingInternalId))
                    {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_)
                {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                }

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
            // copy datapoint attr-mem in
            if (datapoint_attr)
                memcpy(data_attr_memory_ + cur_c * size_data_attr_per_element_, datapoint_attr, size_data_attr_per_element_);

            if (curlevel)
            {
                linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1); // allocate memory for edges (num_edges * num_levels)
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1)
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--)
                    {
                        bool changed = true;
                        while (changed) // navigate nodes in level_ to find the nearest to query (data_point)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level); // in start this would be the ep => we get neighbors
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++) // loop over neighbors of current node
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_); // compute dist between data_point and nei
                                if (d < curdist)                                                                  // if better distance update current_node and dist_ref (curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) // navigate upto level 0 and connect data_point
                {
                    if (level > maxlevelcopy || level < 0) // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                    if (epDeleted)
                    {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            }
            else
            {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy)
            {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            // track time
            int trackInterval = 100000;
            if ((cur_c + 1) % trackInterval == 0)
            {
                auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                auto tm = std::localtime(&now);

                std::cout << "(G construction) Progress: " << (cur_c + 1) << " iterations completed. Current time: "
                          << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec << std::endl;
            }
            return cur_c;
        }
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr, const void *query_data_attr = nullptr, const bool collect_metrics = false) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (cur_element_count == 0)
                return result;

            // int valid_indices_count = getAllValidIndicesCount(query_data_attr, dist_attr_func_param_);
            // int *valid_indices = getAllValidIndices(query_data_attr, dist_attr_func_param_);

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            // char *currObj1AttrAgg = (getDataAttrAggByInternalId(enterpoint_node_));
            // dist_t distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
            // curdist += hybrid_factor_ * distAttrAgg;
            // std::cout << "maxlevel_: " << maxlevel_ << std::endl;
            for (int level = maxlevel_; level > 0; level--)
            {

                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *)get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                        // dist_t distAttrAgg = fstdistattraggfunc_((void *)valid_indices, currObj1AttrAgg, (void *)&valid_indices_count);
                        // d += hybrid_factor_ * distAttrAgg;
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (search_mode == 0)
            {
                std::priority_queue<std::pair<std::pair<dist_t, bool>, tableint>, std::vector<std::pair<std::pair<dist_t, bool>, tableint>>, CompareByFirstBis> top_candidates;
                top_candidates = searchBaseLayerSTAttrOnly<false>(
                    currObj, query_data, std::max(k, ef_), isIdAllowed, query_data_attr, collect_metrics);
                while (top_candidates.size() > k)
                {
                    top_candidates.pop();
                }
                while (top_candidates.size() > 0)
                {
                    std::pair<std::pair<dist_t, bool>, tableint> rez = top_candidates.top();
                    result.push(std::pair<dist_t, labeltype>(rez.first.first, getExternalLabel(rez.second)));
                    top_candidates.pop();
                }
                return result;
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (search_mode == 1)
            {
                top_candidates = searchBaseLayerSTFpq<false>(
                    currObj, query_data, std::max(k, ef_), isIdAllowed, query_data_attr, collect_metrics);
            }

            if (search_mode == 2)
            {
                top_candidates = searchBaseLayerSTSnf<false>(
                    currObj, query_data, std::max(k, ef_), isIdAllowed, query_data_attr, collect_metrics);
            }
            if (search_mode == 3)
            {
                top_candidates = searchBaseLayerSTSnfRobust<false>(
                    currObj, query_data, std::max(k, ef_), isIdAllowed, query_data_attr, collect_metrics);
            }
            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0)
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        }

        void checkIntegrity()
        {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++)
            {
                for (int l = 0; l <= element_levels_[i]; l++)
                {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++)
                    {
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1)
            {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++)
                {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }
    };
} // namespace hnswlib

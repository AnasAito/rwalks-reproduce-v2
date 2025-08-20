#pragma once
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>

namespace hnswlib
{
    template <typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t>
    {
    public:
        char *data_;
        char *data_attr_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;
        size_t size_attr_per_element_;

        size_t data_size_;
        size_t data_attr_size_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        int dim_attr_;
        std::mutex index_lock;

        std::unordered_map<labeltype, size_t> dict_external_to_internal;

        BruteforceSearch(SpaceInterface<dist_t> *s)
            : data_(nullptr),
              data_attr_(nullptr),
              maxelements_(0),
              cur_element_count(0),
              size_per_element_(0),
              size_attr_per_element_(0),
              data_size_(0),
              data_attr_size_(0),
              dist_func_param_(nullptr),
              dim_attr_(0)
        {
        }

        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
            : data_(nullptr),
              data_attr_(nullptr),
              maxelements_(0),
              cur_element_count(0),
              size_per_element_(0),
              size_attr_per_element_(0),
              data_size_(0),
              data_attr_size_(0),
              dist_func_param_(nullptr),
              dim_attr_(0)
        {
            loadIndex(location, s);
        }

        BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements)
        {
            maxelements_ = maxElements;
            dim_attr_ = *((int *)s->get_dist_attr_func_param());
            data_size_ = s->get_data_size();
            data_attr_size_ = s->get_data_attr_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *)malloc(maxElements * size_per_element_);
            if (data_ == nullptr)
                throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data");
            size_attr_per_element_ = data_attr_size_;
            data_attr_ = (char *)malloc(maxElements * size_attr_per_element_);
            if (data_attr_ == nullptr)
                throw std::runtime_error("Not enough memory: BruteforceSearch failed to allocate data_attr");
            cur_element_count = 0;
        }

        ~BruteforceSearch()
        {
            free(data_);
            free(data_attr_);
        }

        void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false, const void *datapoint_attr = nullptr)
        {
            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);

                auto search = dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end())
                {
                    idx = search->second;
                }
                else
                {
                    if (cur_element_count >= maxelements_)
                    {
                        throw std::runtime_error("The number of elements exceeds the specified limit\n");
                    }
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            // int *datapoint_attr_int = (int *)datapoint_attr;

            // for (int i = 0; i < 12; i++)
            // {
            //     std::cout << datapoint_attr_int[i] << " ";
            // }
            // std ::cout << "adding: " << idx << std::endl;
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
            memcpy(data_attr_ + size_attr_per_element_ * idx, datapoint_attr, data_attr_size_);
            // std::cout << "finsihed adding: " << idx << std::endl;
        }

        void removePoint(labeltype cur_external)
        {
            size_t cur_c = dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            labeltype label = *((labeltype *)(data_ + size_per_element_ * (cur_element_count - 1) + data_size_));
            dict_external_to_internal[label] = cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count - 1),
                   data_size_ + sizeof(labeltype));
            cur_element_count--;
        }
        bool is_point_allowed(int offset, int query_label_idx) const
        {
            return *((int *)(data_attr_ + size_attr_per_element_ * offset + query_label_idx * sizeof(int))) == 1;
        }
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr, const void *datapoint_attr = nullptr, const bool collect_metrics = false) const
        {
            assert(k <= cur_element_count);
            std::priority_queue<std::pair<dist_t, labeltype>> topResults;
            if (cur_element_count == 0)
                return topResults;
            int offset = 0;
            int query_label_idx = -1;
            int *datapoint_attr_int = (int *)datapoint_attr;
            for (int i = 0; i < dim_attr_; i++)
            {
                if (datapoint_attr_int[i] == 1)
                {
                    query_label_idx = i;
                    break;
                }
            }
            // std::cout << "query_label_idx = " << query_label_idx << std::endl;
            while (topResults.size() < k)
            {
                if (is_point_allowed(offset, query_label_idx))
                {
                    dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * offset, dist_func_param_);
                    labeltype label = *((labeltype *)(data_ + size_per_element_ * offset + data_size_));
                    topResults.push(std::pair<dist_t, labeltype>(dist, label));
                }
                ++offset;
            }
            dist_t lastdist = topResults.empty() ? std::numeric_limits<dist_t>::max() : topResults.top().first;
            for (int i = offset; i < cur_element_count; i++)
            {
                if (is_point_allowed(i, query_label_idx))
                {
                    dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                    if (dist <= lastdist)
                    {
                        labeltype label = *((labeltype *)(data_ + size_per_element_ * i + data_size_));
                        if ((!isIdAllowed) || (*isIdAllowed)(label))
                        {
                            topResults.push(std::pair<dist_t, labeltype>(dist, label));
                        }
                        if (topResults.size() > k)
                            topResults.pop();

                        if (!topResults.empty())
                        {
                            lastdist = topResults.top().first;
                        }
                    }
                }
            }
            // std ::cout << "topResults size = " << topResults.size() << std::endl;
            return topResults;
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);

            output.write(data_, maxelements_ * size_per_element_);

            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s)
        {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            readBinaryPOD(input, maxelements_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(labeltype);
            data_ = (char *)malloc(maxelements_ * size_per_element_);
            if (data_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

            input.read(data_, maxelements_ * size_per_element_);

            input.close();
        }
    };
} // namespace hnswlib

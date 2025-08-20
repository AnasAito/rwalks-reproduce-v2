
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <sys/time.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <arpa/inet.h>
#include <assert.h> /* assert */
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <limits.h>
#include <math.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath> // for std::mean and std::stdev
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <thread>
#include "utils.cpp"

// Function to get the data root directory relative to this executable
std::string getDataRootDir() {
    // Get the current working directory and assume it's the acorn directory
    // when the executable is run from the acorn directory
    char currentPath[PATH_MAX];
    if (getcwd(currentPath, sizeof(currentPath)) == nullptr) {
        // Fallback to default path if getcwd fails
        return "/Users/mac/dev/rwalks-reproduce/data";
    }

    // Convert to string and go up one level to get to project root, then down
    // to data
    std::string currentDir(currentPath);
    std::string parentDir = currentDir.substr(0, currentDir.find_last_of('/'));
    std::string dataRoot = parentDir + "/data";

    return dataRoot;
}

// Environment variable helper functions
std::string getEnvVar(
        const std::string& key,
        const std::string& defaultValue = "") {
    const char* val = std::getenv(key.c_str());
    return val ? std::string(val) : defaultValue;
}

void loadEnvFile(const std::string& filename = ".env") {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename
                  << " file. Using environment variables or defaults."
                  << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        // Find the = character
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Remove any leading/trailing whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // Set environment variable
            setenv(key.c_str(),
                   value.c_str(),
                   0); // 0 = don't overwrite if already set
        }
    }
    file.close();
}

void write_neighbors_to_binary(
        const faiss::idx_t* nns2,
        int nq,
        int k,
        int ef_iter,
        std::string spec_id) {
    std::string data_root = getDataRootDir();
    std::string sift50k_dir = data_root + "/acorn_data";
    std::ofstream outFile(
            sift50k_dir + "/01_nn_" + std::to_string(ef_iter) + "_" + spec_id +
                    ".bin",
            std::ios::binary);

    if (!outFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return;
    }

    // Write the number of queries (nq) and neighbors per query (k)
    outFile.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    outFile.write(reinterpret_cast<const char*>(&k), sizeof(k));

    // Write only the neighbors (nns2)
    outFile.write(
            reinterpret_cast<const char*>(nns2), sizeof(faiss::idx_t) * nq * k);

    outFile.close();
}

void write_distances_to_binary(
        const float* dist2,
        int nq,
        int k,
        int ef_iter,
        std::string spec_id) {
    std::string data_root = getDataRootDir();
    std::string sift50k_dir = data_root + "/acorn_data";
    std::ofstream outFile(
            sift50k_dir + "/01_nn_dist" + std::to_string(ef_iter) + "_" +
                    spec_id + ".bin",
            std::ios::binary);

    if (!outFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return;
    }

    // Write the number of queries (nq) and neighbors per query (k)
    outFile.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    outFile.write(reinterpret_cast<const char*>(&k), sizeof(k));

    // Write only the neighbors (nns2)
    outFile.write(reinterpret_cast<const char*>(dist2), sizeof(float) * nq * k);

    outFile.close();
}

void write_qps_to_binary(const std::vector<float>& qps_values) {
    std::string data_root = getDataRootDir();
    std::string sift50k_dir = data_root + "/acorn_data";
    std::ofstream outFile(sift50k_dir + "/all_qps.bin", std::ios::binary);

    if (!outFile) {
        std::cerr << "Error opening QPS file for writing" << std::endl;
        return;
    }

    // Write the total number of QPS values
    size_t num_qps = qps_values.size();
    outFile.write(reinterpret_cast<const char*>(&num_qps), sizeof(num_qps));

    // Write all QPS values as a long float array
    outFile.write(
            reinterpret_cast<const char*>(qps_values.data()),
            sizeof(float) * num_qps);

    outFile.close();

    std::cout << "Saved " << num_qps << " QPS values to all_qps.bin"
              << std::endl;
}

// void write_neighbors_to_binary(
//         const faiss::idx_t* nns2,
//         int nq,
//         int k,
//         int ef_iter) {
//     std::ofstream outFile(
//             "/home/anas.aitaomar/ACORN/testing_data/sift50k/nn_" +
//                     std::to_string(ef_iter) + ".bin",
//             std::ios::binary);

//     if (!outFile) {
//         std::cerr << "Error opening file for writing" << std::endl;
//         return;
//     }

//     // Write the number of queries (nq) and neighbors per query (k)
//     outFile.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
//     outFile.write(reinterpret_cast<const char*>(&k), sizeof(k));

//     // Write only the neighbors (nns2)
//     outFile.write(
//             reinterpret_cast<const char*>(nns2), sizeof(faiss::idx_t) * nq *
//             k);

//     outFile.close();
// }

void read_permitted_ids_from_bin(
        const std::string& filename,
        std::vector<char>& filter_ids_map,
        size_t nq,
        size_t N) {
    // Open the binary file in input mode and binary mode
    std::ifstream file(filename, std::ios::in | std::ios::binary);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Resize the vector to the required size (nq * N)
    filter_ids_map.resize(nq * N);

    // Read the entire content of the file into the vector
    file.read(
            reinterpret_cast<char*>(filter_ids_map.data()),
            filter_ids_map.size());

    // Close the file
    file.close();
}

// create indices for debugging, write indices to file, and get recall stats for
// all queries
int main(int argc, char* argv[]) {
    // Load environment variables from .env file
    loadEnvFile();

    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout
            << "====================\nSTART: running TEST_ACORN for hnsw, sift data --"
            << nthreads << "cores\n"
            << std::endl;
    // printf("====================\nSTART: running MAKE_INDICES for hnsw
    // --...\n");
    double t0 = elapsed();

    int efc = 40;   // default is 40
    int efs = 10;   //  default is 16
    int k = 10;     // search parameter
    size_t d = 128; // dimension of the vectors to index - will be overwritten
                    // by the dimension of the dataset
    int M;          // HSNW param M TODO change M back
    int M_beta;     // param for compression
    // float attr_sel = 0.001;
    // int gamma = (int) 1 / attr_sel;
    int gamma;
    //     int n_centroids;
    // int filter = 0;
    std::string dataset; // must be sift1B or sift1M or tripclick
                         //     int test_partitions = 0;
                         //     int step = 10; // 2

    std::string assignment_type = "rand";
    //     int alpha = 0;

    //     srand(0); // seed for random number generator
    //     int num_trials = 60;

    size_t N = 0; // N will be how many we truncate nb from sift1M to

    int opt;
    { // parse arguments

        if (argc < 6 || argc > 8) {
            fprintf(stderr,
                    "Syntax: %s <number vecs> <gamma> [<assignment_type>] [<alpha>] <dataset> <M> <M_beta>\n",
                    argv[0]);
            exit(1);
        }

        N = strtoul(argv[1], NULL, 10);
        printf("N: %ld\n", N);

        gamma = atoi(argv[2]);
        printf("gamma: %d\n", gamma);

        dataset = argv[3];
        printf("dataset: %s\n", dataset.c_str());
        // if (dataset != "sift1M" && dataset != "sift1M_test" &&
        //     dataset != "sift1B" && dataset != "tripclick" &&
        //     dataset != "paper" && dataset != "paper_rand2m") {
        //     printf("got dataset: %s\n", dataset.c_str());
        //     fprintf(stderr,
        //             "Invalid <dataset>; must be a value in [sift1M,
        //             sift1B]\n");
        //     exit(1);
        // }

        M = atoi(argv[4]);
        printf("M: %d\n", M);

        M_beta = atoi(argv[5]);
        printf("M_beta: %d\n", M_beta);
    }

    // load metadata
    //     n_centroids = gamma;

    //     std::vector<int> metadata = load_attrb(dataset, true);
    //     metadata.resize(N);
    //     assert(N == metadata.size());
    //     printf("[%.3f s] Loaded metadata, %ld attr's found\n",
    //            elapsed() - t0,
    //            metadata.size());
    std::vector<int> metadata(N, 0);
    metadata.resize(N);
    assert(N == metadata.size());
    printf("[%.3f s] Loaded metadata, %ld attr's found\n",
           elapsed() - t0,
           metadata.size());

    // size_t nq;
    // float* xq;
    // std::vector<int> aq;
    // { // load query vectors and attributes
    //     printf("[%.3f s] Loading query vectors and attributes\n",
    //            elapsed() - t0);

    //     size_t d2;
    //     // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
    //     bool is_base = 0;
    //     // load_data(dataset, is_base, &d2, &nq, xq);
    //     std::string filename =
    //             "/home/anas.aitaomar/ACORN/testing_data/sift50k/query.fvecs";

    //     xq = fvecs_read(filename.c_str(), &d2, &nq);
    //     assert(d == d2 ||
    //            !"query does not have same dimension as expected 128");
    //     if (d != d2) {
    //         d = d2;
    //     }

    //     std::cout << "query vecs data loaded, with dim: " << d2 << ", nb=" <<
    //     nq
    //               << std::endl;
    //     printf("[%.3f s] Loaded query vectors from %s\n",
    //            elapsed() - t0,
    //            filename.c_str());
    //     aq = load_attrb(dataset, false);
    //     printf("[%.3f s] Loaded %ld %s queries\n",
    //            elapsed() - t0,
    //            nq,
    //            dataset.c_str());
    // }

    // create normal (base) and hybrid index
    printf("[%.3f s] Index Params -- d: %ld, M: %d, N: %ld, gamma: %d\n",
           elapsed() - t0,
           d,
           M,
           N,
           gamma);

    // ACORN-gamma
    faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
    hybrid_index.acorn.efSearch = efs; // default is 16 HybridHNSW.capp
    debug("ACORN index created%s\n", "");

    // ACORN-1
    // faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata, M * 2);
    // hybrid_index_gamma1.acorn.efSearch = efs; // default is 16
    // HybridHNSW.capp

    { // populating the database
        std::cout << "====================Vectors====================\n"
                  << std::endl;
        // printf("====================Vectors====================\n");

        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        // bool is_base = 1;
        // std::string filename = get_file_name(dataset, is_base);
        // std::string filename =
        //         "/data/anas.aitaomar/sigmod25/and_exp/base.fvecs";
        std::string data_root = getDataRootDir();
        std::string filename = data_root + "/acorn_data/base.fvecs";
        float* xb = fvecs_read(filename.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not dim 128 as expected");
        printf("[%.3f s] Loaded base vectors from file: %s\n",
               elapsed() - t0,
               filename.c_str());

        std::cout << "data loaded, with dim: " << d2 << ", nb=" << nb
                  << std::endl;

        printf("[%.3f s] Indexing database, size %ld*%ld from max %ld\n",
               elapsed() - t0,
               N,
               d2,
               nb);

        // index->add(nb, xb);

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        // base_index.add(N, xb);
        // printf("[%.3f s] Vectors added to base index \n", elapsed() - t0);
        // std::cout << "Base index vectors added: " << nb << std::endl;

        hybrid_index.add(N, xb);
        printf("[%.3f s] Vectors added to hybrid index \n", elapsed() - t0);
        std::cout << "Hybrid index vectors added" << nb << std::endl;
        // printf("SKIPPED creating ACORN-gamma\n");

        // hybrid_index_gamma1.add(N, xb);
        // printf("[%.3f s] Vectors added to hybrid index with gamma=1 \n",
        //        elapsed() - t0);
        // std::cout << "Hybrid index with gamma=1 vectors added" << nb
        //           << std::endl;

        delete[] xb;
    }

    // write hybrid index and partition indices to files
    // {
    //     std::cout << "====================Write Index====================\n"
    //               << std::endl;
    // // write hybrid index
    // // std::string filename = "hybrid_index" + dataset + ".index";
    // std::stringstream filepath_stream;
    // if (dataset == "sift1M" || dataset == "sift1B") {
    //     filepath_stream << "./tmp/hybrid_" << (int)(N / 1000 / 1000)
    //                     << "m_nc=" << 2 << "_assignment=" <<
    //                     assignment_type
    //                     << "_alpha=" << 2 << ".json";

    // } else {
    //     filepath_stream << "/data/anas.aitaomar" << "/hybrid" << "_M=" <<
    //     M
    //                     << "_efc" << efc << "_Mb=" << M_beta
    //                     << "_gamma=" << gamma << ".json";
    // }
    // std::string filepath = filepath_stream.str();
    // write_index(&hybrid_index, filepath.c_str());
    // printf("[%.3f s] Wrote hybrid index to file: %s\n",
    //        elapsed() - t0,
    //        filepath.c_str());

    // write hybrid_gamma1 index
    // std::stringstream filepath_stream2;
    // if (dataset == "sift1M" || dataset == "sift1B") {
    //     filepath_stream2 << "./tmp/hybrid_gamma1_" << (int)(N / 1000 /
    //     1000)
    //                      << "m_nc=" << n_centroids
    //                      << "_assignment=" << assignment_type
    //                      << "_alpha=" << alpha << ".json";

    // } else {
    //     filepath_stream2 << "./tmp/" << dataset << "/hybrid" << "_M=" <<
    //     M
    //                      << "_efc" << efc << "_Mb=" << M_beta
    //                      << "_gamma=" << 1 << ".json";
    // }
    // std::string filepath2 = filepath_stream2.str();
    // write_index(&hybrid_index_gamma1, filepath2.c_str());
    // printf("[%.3f s] Wrote hybrid_gamma1 index to file: %s\n",
    //        elapsed() - t0,
    //        filepath2.c_str());

    // { // write base index
    //     std::stringstream filepath_stream;
    //     if (dataset == "sift1M" || dataset == "sift1B") {
    //         filepath_stream << "./tmp/base_" << (int)(N / 1000 / 1000)
    //                         << "m_nc=" << n_centroids
    //                         << "_assignment=" << assignment_type
    //                         << "_alpha=" << alpha << ".json";

    //     } else {
    //         filepath_stream << "./tmp/" << dataset << "/base" << "_M=" <<
    //         M
    //                         << "_efc=" << efc << ".json";
    //     }
    //     std::string filepath = filepath_stream.str();
    //     write_index(&base_index, filepath.c_str());
    //     printf("[%.3f s] Wrote base index to file: %s\n",
    //            elapsed() - t0,
    //            filepath.c_str());
    // }
    // }

    { // print out stats
        // printf("====================================\n");
        // printf("============ BASE INDEX =============\n");
        // printf("====================================\n");
        // base_index.printStats(false);
        printf("====================================\n");
        printf("============ ACORN INDEX =============\n");
        printf("====================================\n");
        hybrid_index.printStats(false);
        const double BYTES_IN_GB = 1.0 / (1024 * 1024 * 1024);
        std::cout << "Size of index " << sizeof(hybrid_index) * BYTES_IN_GB
                  << " GBs\n";
    }

    printf("==============================================\n");
    printf("====================Search Results====================\n");
    printf("==============================================\n");
    // double t1 = elapsed();
    printf("==============================================\n");
    printf("====================Search====================\n");
    printf("==============================================\n");
    double t1 = elapsed();

    // { // searching the base database
    //     printf("====================HNSW INDEX====================\n");
    //     printf("[%.3f s] Searching the %d nearest neighbors "
    //            "of %ld vectors in the index, efsearch %d\n",
    //            elapsed() - t0,
    //            k,
    //            nq,
    //            base_index.hnsw.efSearch);

    //     std::vector<faiss::idx_t> nns(k * nq);
    //     std::vector<float> dis(k * nq);

    //     std::cout << "here1" << std::endl;
    //     std::cout << "nn and dis size: " << nns.size() << " " <<
    //     dis.size()
    //               << std::endl;

    //     double t1 = elapsed();
    //     base_index.search(nq, xq, k, dis.data(), nns.data());
    //     double t2 = elapsed();

    //     printf("[%.3f s] Query results (vector ids, then distances):\n",
    //            elapsed() - t0);

    //     // take max of 5 and nq
    //     int nq_print = std::min(5, (int)nq);
    //     for (int i = 0; i < nq_print; i++) {
    //         printf("query %2d nn's: ", i);
    //         for (int j = 0; j < k; j++) {
    //             // printf("%7ld (%d) ", nns[j + i * k], metadata.size());
    //             printf("%7ld (%d) ", nns[j + i * k], metadata[nns[j + i *
    //             k]]);
    //         }
    //         printf("\n     dis: \t");
    //         for (int j = 0; j < k; j++) {
    //             printf("%7g ", dis[j + i * k]);
    //         }
    //         printf("\n");
    //         // exit(0);
    //     }

    //     printf("[%.3f s] *** Query time: %f\n", elapsed() - t0, t2 - t1);

    //     // print number of distance computations
    //     // printf("[%.3f s] *** Number of distance computations: %ld\n",
    //     //    elapsed() - t0, base_index.ntotal * nq);
    //     std::cout << "finished base index examples" << std::endl;
    // }

    //     { // look at stats
    //         // const faiss::HybridHNSWStats& stats = index.hnsw_stats;
    //         const faiss::HNSWStats& stats = faiss::hnsw_stats;

    //         std::cout
    //                 << "============= BASE HNSW QUERY PROFILING STATS
    //                 ============="
    //                 << std::endl;
    //         printf("[%.3f s] Timing results for search of k=%d nearest
    //         neighbors of nq=%ld vectors in the index\n",
    //                elapsed() - t0,
    //                k,
    //                nq);
    //         std::cout << "n1: " << stats.n1 << std::endl;
    //         std::cout << "n2: " << stats.n2 << std::endl;
    //         std::cout << "n3 (number distance comps at level 0): " <<
    //         stats.n3
    //                   << std::endl;
    //         std::cout << "ndis: " << stats.ndis << std::endl;
    //         std::cout << "nreorder: " << stats.nreorder << std::endl;
    //         printf("average distance computations per query: %f\n",
    //                (float)stats.n3 / stats.n1);
    //     }

    { // searching the hybrid database
        printf("==================== ACORN INDEX ====================\n");
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index, efsearch %d\n",
               elapsed() - t0,
               k,
               //    nq,
               hybrid_index.acorn.efSearch);

        // create filter_ids_map, ie a bitmap of the ids that are in the
        // filter
        // std::vector<int> spec_list = {0};
        // std::vector<std::string> spec_list = {
        //         "1%", "5%", "10%", "20%", "30%", "50%"};
        // std::vector<std::string> spec_list = {"0.1%", "1%", "5%", "10%",
        // "15%"};
        std::vector<std::string> spec_list = {"0", "1", "2", "3", "4", "5"};

        std::vector<float> all_qps_values;

        for (auto spec_id = spec_list.begin(); spec_id != spec_list.end();
             spec_id++) {
            size_t nq;
            float* xq;
            std::vector<int> aq;

            size_t d2;
            // xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
            bool is_base = 0;
            // load_data(dataset, is_base, &d2, &nq, xq);
            //     std::string filename =
            //         "/data/anas.aitaomar/sigmod25/and_exp/queries_spec_id_"
            //         + *spec_id + ".fvecs";
            std::string data_root = getDataRootDir();
            std::string sift50k_dir = data_root + "/acorn_data";
            std::string filename =
                    sift50k_dir + "/query_" + *spec_id + ".fvecs";

            xq = fvecs_read(filename.c_str(), &d2, &nq);
            std::cout << " number of queries = " << nq << std::endl;
            std::cout << " spec_id = " << *spec_id << std::endl;

            double t1_bin_map = elapsed();

            std::vector<char> filter_ids_map;
            std::string bin_path =
                    sift50k_dir + "/filter_ids_map_" + *spec_id + ".bin";
            std::cout << bin_path << std::endl;
            read_permitted_ids_from_bin(bin_path, filter_ids_map, nq, N);

            double t2_bin_map = elapsed();
            printf("Bin map creation: %f\n", t2_bin_map - t1_bin_map);
            std::vector<int> ef_iter_list = {
                    10,  20,  30,  40,  45,  50,  55,  60,  70,   80,   90,
                    120, 130, 140, 200, 300, 400, 600, 900, 1200, 1500, 2000};
            //  std::vector<int> ef_iter_list = {10};
            for (auto _ef_iter = ef_iter_list.begin();
                 _ef_iter != ef_iter_list.end();
                 _ef_iter++) {
                // hybrid_index.acorn.efSearch = 10 + _ef_iter * 50;
                // std::cout << "query with ef= " << 10 + _ef_iter * 50 <<
                // std::endl;
                hybrid_index.acorn.efSearch = *_ef_iter;
                std::cout << "query with ef= " << *_ef_iter << std::endl;
                std::vector<faiss::idx_t> nns2(k * nq);
                std::vector<float> dis2(k * nq);
                double t1_x = elapsed();
                hybrid_index.search(
                        nq,
                        xq,
                        k,
                        dis2.data(),
                        nns2.data(),
                        filter_ids_map.data()); // TODO change first
                                                // argument back to nq
                double t2_x = elapsed();

                printf("[%.3f s] Query results (vector ids, then distances):\n",
                       elapsed() - t0);

                std::cout << "flushing neighbors logs to disk ..." << std::endl;
                write_neighbors_to_binary(
                        nns2.data(), nq, k, *_ef_iter, *spec_id);
                write_distances_to_binary(
                        dis2.data(), nq, k, *_ef_iter, *spec_id);

                printf("[%.3f s] *** Query time: %f\n",
                       elapsed() - t0,
                       t2_x - t1_x);
                printf("[%.3f s] *** Query QPS: %f\n",
                       elapsed() - t0,
                       nq / (t2_x - t1_x));
                all_qps_values.push_back(nq / (t2_x - t1_x));
            }

            std::cout << "finished hybrid index examples" << std::endl;
            std::cout << "-------------------------------" << std::endl;
        }
        write_qps_to_binary(all_qps_values);
    }

    printf("[%.3f s] -----DONE-----\n", elapsed() - t0);
}

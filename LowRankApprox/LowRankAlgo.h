#pragma once

#include<vector>
#include <memory>
#include "neurons/NeuronsExtraInfo.h"

/**Enum for ways to compute the low rank approximation
*/
enum class LowRankAlgoEnum {
    Default,
    ACA,
    NMF,
    Centroid,
    Centroid_weighted,
};

inline std::string stringify(LowRankAlgoEnum algo_enum) {
    if (algo_enum == LowRankAlgoEnum::Default ) {
        return "Default";
    }
    if (algo_enum == LowRankAlgoEnum::ACA ) {
        return "ACA";
    }
    if (algo_enum == LowRankAlgoEnum::NMF ) {
        return "NMF";
    }
    if (algo_enum == LowRankAlgoEnum::Centroid) {
        return "Centroid";
    }
    if (algo_enum == LowRankAlgoEnum::Centroid_weighted) {
        return "Centroid_weighted";
    }
    return "";
}





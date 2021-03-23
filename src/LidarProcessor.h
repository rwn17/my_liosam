#include "ImageProjection.h"
#include "FeatureExtraction.h"
#include "MapOptimization.h"

class LidarProcessor{
    public:

    LidarProcessor()
    {        
        ReadInParams();
        
        usleep(100);
    }

    private:
    void ReadInParams();

    ImageProjection imageProjection;
    FeatureExtraction featureExtraction;
    MapOptimization mapOptimization;
}
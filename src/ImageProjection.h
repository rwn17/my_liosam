#include "utility.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;
const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    double timeScanCur;
    double timeScanEnd;

    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;


public:
    ImageProjection();

    void allocateMemory();

    void resetParameters();

    ~ImageProjection(){};

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg);

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg);

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);

    bool deskewInfo();

    void imuDeskewInfo();

    void odomDeskewInfo();

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur);

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur);

    PointType deskewPoint(PointType *point, double relTime);

    void projectPointCloud();

    void cloudExtraction();
    
    void publishClouds();

    void initializationValue();

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn);

    void calculateSmoothness();

    void markOccludedPoints();

    void extractFeatures();

    void freeCloudInfoMemory();

    void publishFeatureCloud();
};

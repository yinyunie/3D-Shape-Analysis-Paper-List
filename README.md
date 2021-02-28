# 3D-Shape-Analysis-Paper-List
A list of papers, libraries and datasets I recently read is collected for anyone who shows interest at 

---
- [3D Detection](#3d-detection)
- [Shape Representation](#shape-representation)
- [Shape & Scene Completion](#shape--scene-completion)
- [Shape Reconstruction](#shape-reconstruction)
- [3D Scene Understanding](#3d-scene-understanding)
- [3D Scene Reconstruction](#3d-scene-reconstruction)
- [About Human Body](#about-human-body)
- [General Methods](#general-methods)
- [Others (inc. Networks in Classification, Matching, Registration, Alignment, Depth, Normal, Pose, Keypoints, etc.)](#others-inc-networks-in-classification-matching-registration-alignment-depth-normal-pose-keypoints-etc)
- [Survey, Resources and Tools](#survey-resources-and-tools)
---


Statistics: :fire: code is available & stars >= 100 &emsp;|&emsp; :star: citation >= 50




## 3D Detection & Segmentation
- [[Arxiv](https://arxiv.org/pdf/2102.03747.pdf)] DPointNet: A Density-Oriented PointNet for 3D Object Detection in Point Clouds
- [[Arxiv](https://arxiv.org/abs/2102.00463)] PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection [[pytorch](https://github.com/open-mmlab/OpenPCDet)]
- [[Arxiv](https://arxiv.org/pdf/2101.11952v1.pdf)] Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss
- [[Arxiv](https://arxiv.org/pdf/2006.04080v2.pdf)] CubifAE-3D: Monocular Camera Space Cubification for Auto-Encoder based
3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2101.02672.pdf)] Self-Attention Based Context-Aware 3D Object Detection [[pytorch](https://github.com/AutoVision-cloud/SA-Det3D)]
- [[Arxiv](https://arxiv.org/pdf/2012.15712.pdf)] Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.03121.pdf)] It’s All Around You: Range-Guided Cylindrical Network for 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2012.04355.pdf)] 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection [[Project](https://thu17cyz.github.io/3DIoUMatch/)]
- [[Arxiv](https://arxiv.org/pdf/2012.05796.pdf)] Demystifying Pseudo-LiDAR for Monocular 3D Object Detection
- [[3DV2020](https://arxiv.org/pdf/2012.09418.pdf)] PanoNet3D: Combining Semantic and Geometric Understanding for LiDAR Point Cloud Detection
- [[AAAI2021](https://arxiv.org/pdf/2012.10412.pdf)] PC-RGNN: Point Cloud Completion and Graph Neural Network for 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2012.10217.pdf)] SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation
- [[Arxiv](https://arxiv.org/pdf/2012.11409.pdf)] 3D Object Detection with Pointformer
- [[WACV2021](https://arxiv.org/pdf/2011.04841.pdf)] CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection [[pytorch](https://github.com/mrnabati/CenterFusion)]
- [[Arxiv](https://arxiv.org/pdf/2011.10033.pdf)] Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation [[pytorch](https://github.com/xinge008/Cylinder3D)]
- [[Arxiv](https://arxiv.org/pdf/2011.09977.pdf)] Learning to Predict the 3D Layout of a Scene
- [[Arxiv](https://arxiv.org/pdf/2011.12001.pdf)] Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes [[Project](https://github.com/qq456cvb/CanonicalVoting)]
- [[Arxiv](https://arxiv.org/pdf/2011.13328.pdf)] DyCo3D: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution
- [[Arxiv](https://arxiv.org/abs/2011.13628)] Temporal-Channel Transformer for 3D Lidar-Based Video Object Detection in Autonomous Driving
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/f2fc990265c712c49d51a18a32b39f0c-Paper.pdf)] Every View Counts: Cross-View Consistency in 3D Object Detection with Hybrid-Cylindrical-Spherical Voxelization
- [[NeurIPS2020](https://papers.nips.cc/paper/2020/file/9b72e31dac81715466cd580a448cf823-Paper.pdf)] Group Contextual Encoding for 3D Point Clouds [[pytorch](https://github.com/AsahiLiu/PointDetectron)]
- [[Arxiv](https://arxiv.org/pdf/2010.16279.pdf)] 3D Object Recognition By Corresponding and Quantizing Neural 3D Scene Representations [[Project](https://mihirp1998.github.io/project_pages/3dq/)]
- [[Arxiv](https://arxiv.org/pdf/2009.05307.pdf)] A Density-Aware PointRCNN for 3D Objection Detection in Point Clouds
- [[Arxiv](https://arxiv.org/abs/2009.00764)] Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training
- [[ECCV2020](https://arxiv.org/pdf/2008.13748.pdf)] Reinforced Axial Refinement Network for Monocular 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2006.01250.pdf)] RUHSNet: 3D Object Detection Using Lidar Data in Real Time [[pytorch](https://github.com/abhinavsagar/ruhsnet)]
- [[IROS2020](http://www.xinshuoweng.com/papers/AB3DMOT/camera_ready.pdf)] 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics [[Project](http://www.xinshuoweng.com/projects/AB3DMOT/)][[Code](https://github.com/xinshuoweng/AB3DMOT)]
- [[ECCV2020](https://arxiv.org/pdf/2007.13138.pdf)] Virtual Multi-view Fusion for 3D Semantic Segmentation
- [[ACMMM2020](https://arxiv.org/pdf/2007.13970.pdf)] Weakly Supervised 3D Object Detection from Point Clouds
- [[ECCV2020](https://arxiv.org/abs/2007.11901)] Weakly Supervised 3D Object Detection from Lidar Point Cloud [[pytorch](https://github.com/hlesmqh/WS3D)]
- [[ECCV2020](https://arxiv.org/abs/2007.09548)] Kinematic 3D Object Detection in Monocular Video
- [[IROS2020](https://arxiv.org/pdf/2007.09836.pdf)] Object-Aware Centroid Voting for Monocular 3D Object Detection
- [[ECCV2020](https://arxiv.org/pdf/2007.10323.pdf)] Pillar-based Object Detection for Autonomous Driving
- [[Arxiv](https://arxiv.org/pdf/2007.02099.pdf)] Local Grid Rendering Networks for 3D Object Detection in Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2006.01250.pdf)] Learning to Detect 3D Objects from Point Clouds in Real Time
- [[Arxiv](https://arxiv.org/pdf/2006.04043.pdf)] SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html)] PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
- [[CVPR2020](https://arxiv.org/pdf/2005.05125.pdf)] FroDO: From Detections to 3D Objects
- [[CVPR2020](https://arxiv.org/pdf/2004.00543.pdf)] Physically Realizable Adversarial Examples for LiDAR Object Detection
- [[CVPR2020](https://arxiv.org/pdf/2006.04356.pdf)] Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_End-to-End_3D_Point_Cloud_Instance_Segmentation_Without_Detection_CVPR_2020_paper.pdf)] End-to-end 3D Point Cloud Instance Segmentation without Detection
- [[CVPR2020](https://arxiv.org/pdf/2003.00504.pdf)] MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf)] Structure Aware Single-stage 3D Object Detection from Point Cloud
- [[CVPR2020](https://arxiv.org/pdf/1912.04799.pdf)] Learning Depth-Guided Convolutions for Monocular 3D Object Detection [[pytorch](https://github.com/dingmyu/D4LCN)] :fire:
- [[CVPR2020](https://arxiv.org/pdf/1912.04986.pdf)] What You See is What You Get: Exploiting Visibility for 3D Object Detection
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ahmed_Density-Based_Clustering_for_3D_Object_Detection_in_Point_Clouds_CVPR_2020_paper.pdf)] Density Based Clustering for 3D Object Detection in Point Clouds
- [[CVPR2020](https://arxiv.org/pdf/2004.03572.pdf)] Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation
- [[CVPR2020](https://arxiv.org/pdf/2004.03080.pdf)] End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection
- [[CVPR2020](https://arxiv.org/pdf/1912.13192.pdf)] PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
- [[CVPR2020](https://arxiv.org/pdf/2004.05679.pdf)] MLCVNet: Multi-Level Context VoteNet for 3D Object Detection
- [[CVPR2020](https://arxiv.org/pdf/1911.10150.pdf)] PointPainting: Sequential Fusion for 3D Object Detection
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Joint_3D_Instance_Segmentation_and_Object_Detection_for_Autonomous_Driving_CVPR_2020_paper.pdf)] Joint 3D Instance Segmentation and Object Detection for Autonomous Driving
- [[CVPR2020](https://arxiv.org/pdf/2003.01251.pdf)] Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud [[tensorflow](https://github.com/WeijingShi/Point-GNN)]
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Joint_3D_Instance_Segmentation_and_Object_Detection_for_Autonomous_Driving_CVPR_2020_paper.pdf)] Joint 3D Instance Segmentation and Object Detection for Autonomous Driving
- [[CVPR2020](https://arxiv.org/pdf/2003.00186.pdf)] HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Hierarchical_Graph_Network_for_3D_Object_Detection_on_Point_CVPR_2020_paper.pdf)] A Hierarchical Graph Network for 3D Object Detection on Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2006.05682.pdf)] H3DNet: 3D Object Detection Using Hybrid Geometric Primitives
- [[CVPR2020](https://arxiv.org/pdf/2005.13888.pdf)] P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2004.12636.pdf)] 3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection
- [[CVPR2020](https://arxiv.org/pdf/2004.09305.pdf)] Joint Spatial-Temporal Optimization for Stereo 3D Object Tracking
- [[CVPR2020](https://arxiv.org/pdf/2004.08745.pdf)] Learning to Evaluate Perception Models Using Planner-Centric Metrics
- [[CVPR2020](https://arxiv.org/pdf/2004.03572.pdf)] Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation [[pytorch](https://github.com/zju3dv/disprcnn)]
- [[Arxiv](https://arxiv.org/pdf/2004.02774.pdf)] SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds [[github](https://github.com/xinge008/SSN)]
- [[CVPR2020](https://arxiv.org/pdf/2004.03080.pdf)] End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection [[github](https://github.com/mileyan/pseudo-LiDAR_e2e)]
- [[Arxiv](https://arxiv.org/pdf/2004.02693.pdf)] Finding Your (3D) Center: 3D Object Detection Using a Learned Loss
- [[CVPR2020](https://arxiv.org/pdf/2004.01658.pdf)] PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation
- [[CVPR2020](https://arxiv.org/pdf/2003.13867.pdf)] 3D-MPA: Multi Proposal Aggregation for 3D Semantic Instance Segm
- [[CVPR2020](https://arxiv.org/pdf/2003.06233.pdf)] Fusion-Aware Point Convolution for Online Semantic 3D Scene Segmentation
- [[CVPR2020](https://arxiv.org/pdf/2003.06537.pdf)] OccuSeg: Occupancy-aware 3D Instance Segmentation
- [[CVPR2020](https://arxiv.org/pdf/2003.05593.pdf)] Learning to Segment 3D Point Clouds in 2D Image Space
- [[CVPR2020](https://arxiv.org/pdf/2003.01251.pdf)] Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud [[tensorflow](https://github.com/WeijingShi/Point-GNN)]
- [[AAAI2020](https://arxiv.org/pdf/2003.00529.pdf)] ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2003.00504.pdf)] MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships
- [[Arxiv](https://arxiv.org/pdf/2003.00186.pdf)] HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2002.10111.pdf)] SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation
- [[Arxiv](https://arxiv.org/pdf/2002.10187.pdf)] 3DSSD: Point-based 3D Single Stage Object Detector
- [[Arxiv](https://arxiv.org/pdf/2002.01619.pdf)] Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation
- [[CVPR2020](https://arxiv.org/pdf/2001.10692.pdf)] ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes
- [[Arxiv](https://arxiv.org/pdf/2001.10609.pdf)] A Review on Object Pose Recovery: from 3D Bounding Box Detectors to Full 6D Pose Estimators
- [[Arxiv](https://arxiv.org/pdf/1912.08830.pdf)] ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language
- [[Arxiv](https://arxiv.org/pdf/1904.07850.pdf)] Objects as Points [[github](https://github.com/xingyizhou/CenterNet)] :star::fire:
- [[Arxiv](https://arxiv.org/pdf/2001.03343.pdf)] RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving [[github](https://github.com/Banconxuan/RTM3D)]
- [[CVPR2020](https://arxiv.org/pdf/2001.03398.pdf)] DSGN: Deep Stereo Geometry Network for 3D Object Detection [[github](https://github.com/chenyilun95/DSGN)]
- [[Arxiv](https://arxiv.org/pdf/2001.01349.pdf)] Learning and Memorizing Representative Prototypes for 3D Point Cloud Semantic and Instance Segmentation
- [[Arxiv](https://arxiv.org/pdf/1912.13192.pdf)] PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/1912.12791.pdf)] Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots
- [[CVPR2020](https://arxiv.org/pdf/1912.11803.pdf)] SESS: Self-Ensembling Semi-Supervised 3D Object Detection
- [[NeurIPS2019](https://papers.nips.cc/paper/9093-perspectivenet-3d-object-detection-from-a-single-rgb-image-via-perspective-points)] PerspectiveNet: 3D Object Detection from a Single RGB Image via Perspective Points
- [[NeurIPS2019](https://arxiv.org/pdf/1906.01140.pdf)] Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.pdf)] Deep Hough Voting for 3D Object Detection in Point Clouds
- [[AAAI2020](https://arxiv.org/pdf/1912.09654.pdf)] JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Brazil_M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection_ICCV_2019_paper.pdf)] M3D-RPN: Monocular 3D Region Proposal Network for Object Detection [[pytorch](https://github.com/garrickbrazil/M3D-RPN)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)] 3D Instance Segmentation via Multi-Task Metric Learning
- [[Arxiv](https://arxiv.org/pdf/1912.08035.pdf)] Single-Stage Monocular 3D Object Detection with Virtual Cameras
- [[Arxiv](https://arxiv.org/pdf/1912.10336.pdf)] Depth Completion via Deep Basis Fitting
- [[Arxiv](https://arxiv.org/pdf/1912.00202.pdf)] Relation Graph Network for 3D Object Detection in Point Clouds
- [[CVPR2019](https://arxiv.org/pdf/1812.07003.pdf)] 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans [[pytorch](https://github.com/Sekunde/3D-SIS)] :fire:
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.pdf)] Rescan: Inductive Instance Segmentation for Indoor RGBD Scans [[C++](https://github.com/mhalber/Rescan)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Tang_Transferable_Semi-Supervised_3D_Object_Detection_From_RGB-D_Data_ICCV_2019_paper.pdf)] Transferable Semi-Supervised 3D Object Detection From RGB-D Data
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_STD_Sparse-to-Dense_3D_Object_Detector_for_Point_Cloud_ICCV_2019_paper.pdf)] STD: Sparse-to-Dense 3D Object Detector for Point Cloud
- [[CVPR2019](https://arxiv.org/pdf/1812.04244.pdf)] PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud [[pytorch](https://github.com/sshaoshuai/PointRCNN)]
- [[Arxiv](https://arxiv.org/pdf/1908.02990.pdf)] Fast Point R-CNN
- [[Arxiv](https://arxiv.org/pdf/1908.09492.pdf)] Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection [[pytorch](https://github.com/poodarchu/Det3D)] :fire:
- [[ECCV2018](https://arxiv.org/pdf/1803.10409.pdf)] 3DMV: Joint 3D-Multi-View Prediction for 3D Semantic Scene Segmentation [[pytorch](https://github.com/angeladai/3DMV)] :fire:







---
## Shape Representation
- [[Arxiv](https://arxiv.org/abs/2102.09105)] DeepMetaHandles: Learning Deformation Meta-Handles of 3D Meshes with Biharmonic Coordinates [[Project](https://github.com/Colin97/DeepMetaHandles)]

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.00230.pdf)] Point2Skeleton: Learning Skeletal Representations from Point Clouds [[pytorch](https://github.com/clinplayer/Point2Skeleton)]
- [[Arxiv](https://arxiv.org/pdf/2012.03028.pdf)] ParaNet: Deep Regular Representation for 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2012.05657.pdf)] Geometric Adversarial Attacks and Defenses on 3D Point Clouds [[tensorflow](https://github.com/itailang/geometric_adv)]
- [[Arxiv](https://arxiv.org/pdf/2012.07290.pdf)] Learning Category-level Shape Saliency via Deep Implicit Surface Networks
- [[Arxiv](https://arxiv.org/pdf/2012.00926.pdf)] pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis
- [[Arxiv](https://arxiv.org/pdf/2011.14565.pdf)] Deep Implicit Templates for 3D Shape Representation
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/731c83db8d2ff01bdc000083fd3c3740-Paper.pdf)] MetaSDF: Meta-learning Signed Distance Functions [[Project](https://vsitzmann.github.io/metasdf/)]
- [[Arxiv](https://arxiv.org/abs/2010.00973)] RISA-Net: Rotation-Invariant Structure-Aware Network for Fine-Grained 3D Shape Retrieval [[tensorflow](https://github.com/IGLICT/RisaNET)]
- [[Arxiv](https://arxiv.org/abs/2009.09808)] Overfit Neural Networks as a Compact Shape Representation
- [[Arxiv](https://arxiv.org/abs/2008.05440)] DSM-Net: Disentangled Structured Mesh Net for Controllable Generation of Fine Geometry [[Project](http://geometrylearning.com/dsm-net/)]
- [[Arxiv](https://arxiv.org/pdf/2008.01639.pdf)] PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations
- [[Arxiv](https://arxiv.org/pdf/2008.02792.pdf)] CaSPR: Learning Canonical Spatiotemporal Point Cloud Representations
- [[Arxiv](https://arxiv.org/pdf/2008.03875.pdf)] ROCNET: RECURSIVE OCTREE NETWORK FOR EFFICIENT 3D DEEP REPRESENTATION
- [[ECCV2020](https://arxiv.org/pdf/2008.04852.pdf)] GeLaTO: Generative Latent Textured Objects [[Project](https://gelato-paper.github.io/)]
- [[ECCV2020](https://arxiv.org/abs/2007.13393)] Ladybird: Quasi-Monte Carlo Sampling for Deep Implicit Field Based 3D Reconstruction with Symmetry
- [[Arxiv](https://arxiv.org/pdf/2007.11571.pdf)] Neural Sparse Voxel Fields
- [[CVPR2020](https://arxiv.org/pdf/2004.09995.pdf)] StructEdit: Learning Structural Shape Variations [[github](https://github.com/hyzcn/structedit)]
- [[Arxiv](https://arxiv.org/pdf/2004.09995.pdf)] PAI-GCN: Permutable Anisotropic Graph Convolutional Networks for 3D Shape Representation Learning [[github](https://github.com/Gaozhongpai/PaiConvMesh)]
- [[CVPR2020](https://arxiv.org/pdf/2004.03028.pdf)] Learning Generative Models of Shape Handles [[Project page](http://mgadelha.me/shapehandles/)]
- [[CVPR2020](https://arxiv.org/pdf/2004.02869.pdf)] DualSDF: Semantic Shape Manipulation using a Two-Level Representation [[github](https://github.com/zekunhao1995/DualSDF)]
- [[CVPR2020](https://arxiv.org/pdf/2004.01176.pdf)] Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image [[pytorch](https://github.com/paschalidoud/hierarchical_primitives)]
- [[NeurIPS2019](https://arxiv.org/pdf/1906.01618.pdf)] Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations [[pytorch](https://github.com/vsitzmann/scene-representation-networks)]
- [[Arxiv](https://arxiv.org/pdf/2003.13834.pdf)] Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions
- [[Arxiv](https://arxiv.org/pdf/2003.12971.pdf)] Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2003.10983.pdf)] Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2003.05559.pdf)] SeqXY2SeqZ: Structure Learning for 3D Shapes by Sequentially Predicting 1D Occupancy Segments From 2D Coordinates
- [[CVPR2020](https://arxiv.org/pdf/2003.03164.pdf)] D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features
- [[Arxiv](https://arxiv.org/pdf/2002.10099.pdf)] Implicit Geometric Regularization for Learning Shapes
- [[Arxiv](https://arxiv.org/pdf/2002.06597.pdf)] Analytic Marching: An Analytic Meshing Solution from Deep Implicit Surface Networks
- [[Arxiv](https://arxiv.org/pdf/2002.00349.pdf)] Adversarial Generation of Continuous Implicit Shape Representations [[pytorch](https://github.com/marian42/shapegan)]
- [[Arxiv](https://arxiv.org/pdf/2001.02823.pdf)] A Novel Tree-structured Point Cloud Dataset For Skeletonization Algorithm Evaluation [[dataset](https://github.com/liujiboy/TreePointCloud)]
- [[CVPRW2019](http://openaccess.thecvf.com/content_CVPRW_2019/papers/SkelNetOn/Demir_SkelNetOn_2019_Dataset_and_Challenge_on_Deep_Learning_for_Geometric_CVPRW_2019_paper.pdf)] SkelNetOn 2019: Dataset and Challenge on Deep Learning for Geometric Shape Understanding [[project](http://ubee.enseeiht.fr/skelneton/)]
- [[Arxiv](https://arxiv.org/pdf/1912.11932.pdf)] Skeleton Extraction from 3D Point Clouds by Decomposing the Object into Parts
- [[Arxiv](https://arxiv.org/pdf/1912.11606.pdf)] InSphereNet: a Concise Representation and Classification Method for 3D Object
- [[Arxiv](https://arxiv.org/pdf/1912.06126v1.pdf)] Deep Structured Implicit Functions
- [[CVIU](https://reader.elsevier.com/reader/sd/pii/S1077314218303606?token=0CC172174E5193815DEF57234C50AD55CFA60AAB3672EAC166AEFF051C2021E08D78D78CC1A4716A2317128070FF756C)] 3D articulated skeleton extraction using a single consumer-grade depth camera
- [[ICLR2019](https://arxiv.org/pdf/1810.05795.pdf)] Point Cloud GAN [[tensorflow](https://github.com/chunliangli/Point-Cloud-GAN)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Genova_Learning_Shape_Templates_With_Structured_Implicit_Functions_ICCV_2019_paper.pdf)] Learning Shape Templates with Structured Implicit Functions
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shu_3D_Point_Cloud_Generative_Adversarial_Network_Based_on_Tree_Structured_ICCV_2019_paper.pdf)] 3D Point Cloud Generative Adversarial Network Based on
Tree Structured Graph Convolutions [[pytorch](https://github.com/seowok/TreeGAN)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Michalkiewicz_Implicit_Surface_Representations_As_Layers_in_Neural_Networks_ICCV_2019_paper.pdf)] Implicit Surface Representations as Layers in Neural Networks
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)] DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation [[pytorch](https://github.com/facebookresearch/DeepSDF)] :fire: :star:
- [[SIGGRAPH2019](https://arxiv.org/pdf/1908.00575.pdf)] StructureNet: Hierarchical Graph Networks for 3D Shape Generation [[pytorch](https://github.com/daerduoCarey/structurenet)]
- [[SIGGRAPH Asia2019](https://arxiv.org/pdf/1903.10170.pdf)] LOGAN: Unpaired Shape Transform in Latent Overcomplete Space [[tensorflow](https://github.com/kangxue/LOGAN)]
- [[TOG](https://www.researchgate.net/profile/Yajie_Yan2/publication/326726499_Voxel_cores_efficient_robust_and_provably_good_approximation_of_3D_medial_axes/links/5b7712cd4585151fd11b316a/Voxel-cores-efficient-robust-and-provably-good-approximation-of-3D-medial-axes.pdf)] Voxel Cores: Efficient, robust, and provably good approximation of 3D medial axes
- [[SIGGRAPH2018](https://arxiv.org/pdf/1803.09263.pdf)] P2P-NET: Bidirectional Point Displacement Net for Shape Transform [[tensorflow](https://github.com/kangxue/P2P-NET)]
- [[ICML2018](https://arxiv.org/pdf/1707.02392.pdf)] Learning Representations and Generative Models for 3D Point Clouds [[tensorflow](https://github.com/optas/latent_3d_points)] :fire::star:
- [[NeurIPS2018](https://arxiv.org/pdf/1807.03146.pdf)] Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning [[tensorflow](https://github.com/tensorflow/models/tree/master/research/keypointnet)][[project page](https://keypointnet.github.io/)]:star::fire:
- [[AAAI2018](http://graphics.cs.uh.edu/wp-content/papers/2018/2018-AAAI-SkeletonExtractionFromDepthCamera.pdf)] Unsupervised Articulated Skeleton Extraction from Point Set Sequences Captured by a Single Depth Camera
- [[3DV2018](https://arxiv.org/pdf/1808.01337.pdf)] Parsing Geometry Using Structure-Aware Shape Templates
- [[SIGGRAPH2017](https://www.cse.iitb.ac.in/~sidch/docs/siggraph2017_grass.pdf)] GRASS: Generative Recursive Autoencoders for Shape Structures [[pytorch](https://github.com/junli-lj/Grass)] :fire:
- [[TOG](https://pdfs.semanticscholar.org/cc9a/2d7aff3a4238812c29e7d7525b4e4794fffc.pdf)] Erosion Thickness on Medial Axes of 3D Shapes
- [[Vis Comput](https://link.springer.com/content/pdf/10.1007%2Fs00371-016-1331-z.pdf)] Distance field guided L1-median skeleton extraction
- [[CGF](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.13098)] Contracting Medial Surfaces Isotropically for Fast Extraction of Centred Curve Skeletons
- [[CGF](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.13570)] Improved Use of LOP for Curve Skeleton Extraction
- [[SIGGRAPH Asia2015](https://boris.unibe.ch/81116/1/dpoints.pdf)] Deep Points Consolidation [[C++ & Qt](https://www.dropbox.com/s/hroijgjajj4cadi/point-cloud-processing-vs2013-201908.zip?dl=0)]
- [[SIGGRAPH2015](https://dl.acm.org/doi/pdf/10.1145/2787626.2792658)] Burning The Medial Axis
- [[SIGGRAPH2009](http://www-evasion.imag.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2010/Bib/tagliasacchi_zhang_cohen-or-siggraph2009.pdf)] Curve Skeleton Extraction from Incomplete Point Cloud [[matlab](https://github.com/ataiya/rosa)] :star:
- [[TOG](https://arxiv.org/pdf/1908.04520.pdf)] SDM-NET: deep generative network for structured deformable mesh
- [[TOG](https://dl.acm.org/doi/pdf/10.1145/2601097.2601161?download=true)] Robust and Accurate Skeletal Rigging from Mesh Sequences :fire:
- [[TOG](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.729&rep=rep1&type=pdf)] L<sub>1</sub>-medial skeleton of point cloud [[C++](https://github.com/HongqiangWei/L1-Skeleton)] :fire:
- [[EUROGRAPHICS2016](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.12865)] 3D Skeletons: A State-of-the-Art Report :fire:
- [[SGP2012](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.261.3158&rep=rep1&type=pdf)] Mean Curvature Skeletons [[C++](https://github.com/ataiya/starlab-mcfskel)] :fire:
- [[SMIC2010](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5521461)] Point Cloud Skeletons via Laplacian-Based Contraction [[Matlab](https://github.com/ataiya/cloudcontr)] :fire:








---
## Shape & Scene Completion
#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.03408.pdf)] PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths
- [[Arxiv](https://arxiv.org/pdf/2012.09242.pdf)] S3CNet: A Sparse Semantic Scene Completion Network for LiDAR Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2011.09141.pdf)] Semantic Scene Completion using Local Deep Implicit Functions on LiDAR Data
- [[Arxiv](https://arxiv.org/pdf/2011.03981.pdf)] Learning-based 3D Occupancy Prediction for Autonomous Navigation in Occluded Environments
- [[Arxiv](https://arxiv.org/pdf/2012.03408.pdf)] PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths
- [[3DV2020](https://arxiv.org/pdf/2010.13662.pdf)] SCFusion: Real-time Incremental Scene Reconstruction
with Semantic Completion
- [[Arxiv](https://arxiv.org/pdf/2010.04278.pdf)] Refinement of Predicted Missing Parts Enhance Point Cloud
Completion [[pytorch](https://github.com/ivansipiran/Refinement-Point-Cloud-Completion)]
- [[Arxiv](https://arxiv.org/pdf/2009.05290.pdf)] Unsupervised Partial Point Set Registration via
Joint Shape Completion and Registration
- [[Arxiv](https://arxiv.org/abs/2008.10559)] LMSCNet: Lightweight Multiscale 3D Semantic Completion [[Demo](https://www.youtube.com/watch?v=XuEz0mbv2IQ&feature=youtu.be)]
- [[ECCV2020](https://arxiv.org/abs/2008.07358v1)] SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification
- [[ECCV2020](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500273.pdf)] Weakly-supervised 3D Shape Completion in the Wild
- [[Arxiv](https://arxiv.org/pdf/2008.00394.pdf)] Point Cloud Completion by Learning Shape Priors
- [[Arxiv](https://arxiv.org/pdf/2008.00096.pdf)] KAPLAN: A 3D Point Descriptor for Shape Completion
- [[Arxiv](https://arxiv.org/pdf/2008.03404.pdf)] VPC-Net: Completion of 3D Vehicles from MLS Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2006.14660.pdf)] SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/2006.03761.pdf)] GRNet: Gridding Residual Network for Dense Point Cloud Completion
- [[Arxiv](https://arxiv.org/pdf/2006.03762.pdf)] Deep Octree-based CNNs with Output-Guided Skip Connections
for 3D Shape and Scene Completion
- [[CVPR2020](https://arxiv.org/pdf/2005.03871.pdf)] Point Cloud Completion by Skip-attention Network with Hierarchical Folding
- [[CVPR2020](https://arxiv.org/pdf/2004.03327.pdf)] Cascaded Refinement Network for Point Cloud Completion [[github](https://github.com/xiaogangw/cascaded-point-completion)]
- [[CVPR2020](https://arxiv.org/pdf/2004.02122.pdf)] Anisotropic Convolutional Networks for 3D Semantic Scene Completion [[github](https://github.com/waterljwant/SSC)]
- [[AAAI2020](https://arxiv.org/pdf/2003.13910.pdf)] Attention-based Multi-modal Fusion Network for Semantic Scene Completion
- [[CVPR2020](https://arxiv.org/pdf/2003.14052.pdf)] 3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior [[github](https://github.com/charlesCXK/3D-SketchAware-SSC)]
- [[ECCV2020](https://arxiv.org/pdf/2003.07717.pdf)] Multimodal Shape Completion via Conditional Generative Adversarial Networks [[pytorch](https://github.com/ChrisWu1997/Multimodal-Shape-Completion)]
- [[CVPR2020](https://arxiv.org/pdf/1904.12012.pdf)] RevealNet: Seeing Behind Objects in RGB-D Scans
- [[CVPR2020](https://arxiv.org/pdf/2003.01456.pdf)] Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion
- [[CVPR2020](https://arxiv.org/pdf/2003.00410.pdf)] PF-Net: Point Fractal Network for 3D Point Cloud Completion
- [[Arxiv](https://arxiv.org/pdf/2002.07269.pdf)] 3D Gated Recurrent Fusion for Semantic Scene Completion
- [[ICCVW2019](https://arxiv.org/abs/1901.00212)] EdgeConnect: Structure Guided Image Inpainting using Edge Prediction [[pytorch](https://github.com/knazeri/edge-connect)] :fire::star:
- [[ICRA2020](https://arxiv.org/pdf/2001.10709.pdf)] Depth Based Semantic Scene Completion with Position Importance Aware Loss
- [[CVPR2020](https://arxiv.org/pdf/1912.00036.pdf)] SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/1911.10949.pdf)] PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes
- [[ICLR2020](https://arxiv.org/pdf/1904.00069.pdf)] Unpaired Point Cloud Completion on Real Scans using Adversarial Training [[tensorflow](https://github.com/xuelin-chen/pcl2pcl-gan-pub)]
- [[AAAI2020](http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf)] Morphing and Sampling Network for Dense Point Cloud Completion [[pytorch](https://github.com/Colin97/MSN-Point-Cloud-Completion)]
- [[ICCVW2019](http://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Hu_Render4Completion_Synthesizing_Multi-View_Depth_Maps_for_3D_Shape_Completion_ICCVW_2019_paper.pdf)] Render4Completion: Synthesizing Multi-View Depth Maps for 3D Shape
Completion
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ForkNet_Multi-Branch_Volumetric_Semantic_Completion_From_a_Single_Depth_Image_ICCV_2019_paper.pdf)] ForkNet: Multi-branch Volumetric Semantic Completion
from a Single Depth Image [[tensorflow](https://github.com/wangyida/forknet)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Cascaded_Context_Pyramid_for_Full-Resolution_3D_Semantic_Scene_Completion_ICCV_2019_paper.pdf)] Cascaded Context Pyramid for Full-Resolution 3D Semantic Scene Completion [[Caffe3D](https://github.com/Pchank/CCPNet)]
- [[ICCV2019](https://arxiv.org/pdf/1907.12704.pdf)] Multi-Angle Point Cloud-VAE: Unsupervised Feature Learning for 3D Point Clouds from Multiple Angles by Joint Self-Reconstruction and Half-to-Half Prediction
- [[Arxiv](https://arxiv.org/pdf/1908.02893.pdf)] EdgeNet: Semantic Scene Completion from RGB-D images
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)] TopNet: Structural Point Cloud Decoder [[pytorch & tensorflow](https://github.com/lynetcha/completion3d)]
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Han_Deep_Reinforcement_Learning_of_Volume-Guided_Progressive_View_Inpainting_for_3D_CVPR_2019_paper.pdf)] Deep Reinforcement Learning of Volume-guided Progressive View Inpainting for 3D Point Scene Completion from a Single Depth Image
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Giancola_Leveraging_Shape_Completion_for_3D_Siamese_Tracking_CVPR_2019_paper.pdf)] Leveraging Shape Completion for 3D Siamese Tracking [[pytorch](https://github.com/SilvioGiancola/ShapeCompletion3DTracking)]
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sarmad_RL-GAN-Net_A_Reinforcement_Learning_Agent_Controlled_GAN_Network_for_Real-Time_CVPR_2019_paper.pdf)] RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion [[pytorch](https://github.com/iSarmad/RL-GAN-Net)]
- [[3DV2018](https://arxiv.org/pdf/1808.00671.pdf)] PCN: Point Completion Network [[tensorflow](https://github.com/wentaoyuan/pcn)] :fire:
- [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiahui_Zhang_Efficient_Semantic_Scene_ECCV_2018_paper.pdf)] Efficient Semantic Scene Completion Network with Spatial Group Convolution [[pytorch](https://github.com/zjhthu/SGC-Release)]
- [[CVPR2018](https://arxiv.org/pdf/1712.10215.pdf)] ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans [[tensorflow](https://github.com/angeladai/ScanComplete)] :fire::star:
- [[CVPR2018](http://www.cvlibs.net/publications/Stutz2018CVPR.pdf)] Learning 3D Shape Completion from Laser Scan Data with Weak Supervision [[torch](https://github.com/davidstutz/cvpr2018-shape-completion)][[torch](https://github.com/davidstutz/daml-shape-completion)]
- [[IJCV2018](https://arxiv.org/abs/1805.07290)] Learning 3D Shape Completion under Weak Supervision [[torch](https://github.com/davidstutz/aml-improved-shape-completion)][[torch](https://github.com/davidstutz/ijcv2018-improved-shape-completion)]
- [[ICCV2017](https://arxiv.org/pdf/1709.07599.pdf)] High-Resolution Shape Completion Using Deep Neural Networks for Global Structure and Local Geometry Inference :star:
- [[ICCV2017](https://arxiv.org/pdf/1612.00101.pdf)] Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis [[torch](https://github.com/angeladai/cnncomplete)] :fire::star:
- [[CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Semantic_Scene_Completion_CVPR_2017_paper.pdf)] Semantic Scene Completion from a Single Depth Image [[caffe](https://github.com/shurans/sscnet)] :fire::star:
- [[CVPR2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Firman_Structured_Prediction_of_CVPR_2016_paper.pdf)] Structured Prediction of Unobserved Voxels From a Single Depth Image [[resource](http://visual.cs.ucl.ac.uk/pubs/depthPrediction/)] :star:






---
## Shape Reconstruction
- [[Arxiv](https://arxiv.org/abs/2102.08860)] ShaRF: Shape-conditioned Radiance Fields from a Single View [[Project](http://www.krematas.com/sharf/index.html)]
- [[Arxiv](https://arxiv.org/pdf/2102.06195.pdf)] Shelf-Supervised Mesh Prediction in the Wild
- [[Arxiv](https://arxiv.org/pdf/2102.05973.pdf)] HyperPocket: Generative Point Cloud Completion
- [[Arxiv](https://arxiv.org/pdf/2102.02798.pdf)] Im2Vec: Synthesizing Vector Graphics without Vector Supervision [[resource](http://geometry.cs.ucl.ac.uk/projects/2020/Im2Vec/)]
- [[Arxiv](https://arxiv.org/pdf/2101.06860.pdf)] Secrets of 3D Implicit Object Shape Reconstruction in the Wild
- [[Arxiv](https://arxiv.org/pdf/2101.07889.pdf)] Joint Learning of 3D Shape Retrieval and Deformation
- [[Arxiv](https://arxiv.org/abs/2101.10994)] Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.01203.pdf)] Learning Delaunay Surface Elements for Mesh Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2012.02493.pdf)] Compositionally Generalizable 3D Structure Prediction
- [[Arxiv](https://arxiv.org/pdf/2012.03196.pdf)] Online Adaptation for Consistent Mesh Reconstruction in the Wild
- [[Arxiv](https://arxiv.org/pdf/2012.07498.pdf)] Sign-Agnostic Implicit Learning of Surface Self-Similarities for Shape Modeling and Reconstruction from Raw Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2012.07241.pdf)] Deep Optimized Priors for 3D Shape Modeling and Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2011.00844.pdf)] DO 2D GANS KNOW 3D SHAPE? UNSUPERVISED 3D SHAPE RECONSTRUCTION FROM 2D IMAGE GANS [[Project](https://xingangpan.github.io/projects/GAN2Shape.html)]
- [[Arxiv](https://arxiv.org/pdf/2011.02570.pdf)] DUDE: Deep Unsigned Distance Embeddings for Hi-Fidelity Representation of Complex 3D Surfaces
- [[3DV2020](https://arxiv.org/pdf/2011.04755.pdf)] Learning to Infer Semantic Parameters for 3D Shape Editing [[Project](https://github.com/weify627/learn-sem-param)]
- [[3DV2020](https://arxiv.org/pdf/2011.08026.pdf)] Cycle-Consistent Generative Rendering for 2D-3D Modality Translation [[Project](https://ttaa9.github.io/genren/)]
- [[3DV2020](https://arxiv.org/pdf/2011.08534.pdf)] A Divide et Impera Approach for 3D Shape Reconstruction from Multiple Views
- [[Arxiv](https://arxiv.org/pdf/2011.11567.pdf)] A Closed-Form Solution to Local Non-Rigid Structure-from-Motion
- [[Arxiv](https://arxiv.org/pdf/2011.13650.pdf)] Deformed Implicit Field: Modeling 3D Shapes with Learned Dense Correspondence
- [[Arxiv](https://arxiv.org/pdf/2011.13961.pdf)] D-NeRF: Neural Radiance Fields for Dynamic Scenes
- [[Arxiv](https://arxiv.org/pdf/2011.03277.pdf)] Modular Primitives for High-Performance Differentiable Rendering
- [[Arxiv](https://arxiv.org/pdf/2011.14791.pdf)] NeuralFusion: Online Depth Fusion in Latent Space
- [[Arxiv](https://arxiv.org/abs/2012.12247)] Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video [[Project](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/)]
- [[NeurIPS2020](https://arxiv.org/abs/2007.15627)] Continuous Object Representation Networks: Novel View Synthesis without Target View Supervision [[Project](https://nicolaihaeni.github.io/corn/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/83fa5a432ae55c253d0e60dbfa716723-Paper.pdf)] SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images [[Project](https://chenhsuanlin.bitbucket.io/signed-distance-SRN/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/1a77befc3b608d6ed363567685f70e1e-Paper.pdf)] Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance [[Project](https://lioryariv.github.io/idr/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/098d86c982354a96556bd861823ebfbd-Paper.pdf)] Convolutional Generation of Textured 3D Meshes [[Project](https://github.com/dariopavllo/convmesh)]
- [[Arxiv](https://arxiv.org/pdf/2012.04641.pdf)] Vid2CAD: CAD Model Alignment using Multi-View Constraints from Videos
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/21327ba33b3689e713cdff1641128004-Paper.pdf)] UCLID-Net: Single View Reconstruction in Objec Space [[Project](https://github.com/cvlab-epfl/UCLID-Net)]
- [[NeurIPS2020](https://arxiv.org/abs/2008.02792)] CaSPR: Learning Canonical Spatiotemporal Point Cloud Representations
 [[Project](https://geometry.stanford.edu/projects/caspr/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf)] Generative 3D Part Assembly via Dynamic Graph Learning [[pytorch](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/7137debd45ae4d0ab9aa953017286b20-Paper.pdf)] Learning Deformable Tetrahedral Meshes for 3D Reconstruction [[Project](https://nv-tlabs.github.io/DefTet/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf)] SoftFlow: Probabilistic Framework for Normalizing Flow on Manifolds [[pytorch](https://github.com/ANLGBOY/SoftFlow)]
- [[Arxiv](https://arxiv.org/pdf/2010.08276.pdf)] Training Data Generating Networks: Linking 3D Shapes and Few-Shot Classification
- [[Arxiv](https://arxiv.org/pdf/2010.08682.pdf)] MESHMVS: MULTI-VIEW STEREO GUIDED MESH RECONSTRUCTION
- [[Arxiv](https://arxiv.org/pdf/2010.11378.pdf)] Learning Occupancy Function from Point Clouds for Surface Reconstruction
- [[NeurIPS2020](https://arxiv.org/pdf/2010.10505.pdf)] SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images [[Project](https://chenhsuanlin.bitbucket.io/signed-distance-SRN/)]
- [[Arxiv](https://arxiv.org/abs/2010.04595)] GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering [[github](https://github.com/alextrevithick/GRF)]
- [[3DV2020](https://arxiv.org/pdf/2010.05391.pdf)] A Progressive Conditional Generative Adversarial Network
for Generating Dense and Colored 3D Point Clouds
- [[3DV2020](https://arxiv.org/pdf/2010.07021.pdf)] Better Patch Stitching for Parametric Surface Reconstruction
- [[NeurIPS2020](https://arxiv.org/pdf/2010.07428.pdf)] Skeleton-bridged Point Completion: From Global
Inference to Local Adjustment [[Project Page](https://yinyunie.github.io/SKPCN-page/)]
- [[Arxiv](https://arxiv.org/abs/2010.07492)] NeRF++: Analyzing and Improving Neural Radiance Fields [[pytorch](https://github.com/Kai-46/nerfplusplus)]
- [[Arxiv](https://arxiv.org/pdf/2009.03298.pdf)] Improved Modeling of 3D Shapes with Multi-view Depth Maps
- [[SIGGRAPH2020](https://arxiv.org/abs/2008.12298)] One Shot 3D Photography [[Project](https://facebookresearch.github.io/one_shot_3d_photography/)]
- [[BMVC2020](https://arxiv.org/pdf/2008.11762.pdf)] Large Scale Photometric Bundle Adjustment
- [[ECCV2020](https://arxiv.org/abs/2008.10719)] Interactive Annotation of 3D Object Geometry using 2D Scribbles [[Project](http://www.cs.toronto.edu/~shenti11/scribble3d/)]
- [[BMVC2020](https://arxiv.org/abs/2008.07928)] Visibility-aware Multi-view Stereo Network
- [[ECCV2020](https://arxiv.org/abs/2008.07760)] Pix2Surf: Learning Parametric 3D Surface Models of Objects from Images
- [[ECCV2020](https://arxiv.org/abs/2008.06133)] 3D Bird Reconstruction: a Dataset, Model, and Shape Recovery from a Single View [[Project](https://marcbadger.github.io/avian-mesh/)][[Pytorch](https://github.com/marcbadger/avian-mesh)]
- [[BMVC2020](https://arxiv.org/abs/1912.04663)] 3D-GMNet: Single-View 3D Shape Recovery as A Gaussian Mixture
- [[SIGGRAPH2020](https://arxiv.org/pdf/2008.06471.pdf)] Self-Sampling for Neural Point Cloud Consolidation
- [[ECCV2020](https://arxiv.org/abs/2008.00446)] Stochastic Bundle Adjustment for Efficient and Scalable 3D Reconstruction [[github](https://github.com/zlthinker/STBA)]
- [[Arxiv](https://arxiv.org/abs/2008.02268)] NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections [[Project](https://nerf-w.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2005.11617.pdf)] MeshODE: A Robust and Scalable Framework for Mesh Deformation
- [[Arxiv](https://arxiv.org/pdf/2007.12944.pdf)] MRGAN: Multi-Rooted 3D Shape Generation with Unsupervised Part Disentanglement
- [[ECCV2020](https://arxiv.org/pdf/2007.09267.pdf)] Meshing Point Clouds with Predicted Intrinsic-Extrinsic Ratio Guidance [[pytorch](https://github.com/Colin97/Point2Mesh)]
- [[ECCV2020](https://arxiv.org/pdf/2007.11110.pdf)] Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop
- [[ECCV2020](https://arxiv.org/pdf/2007.10872.pdf)] Dense Hybrid Recurrent Multi-view Stereo Net with Dynamic Consistency Checking
- [[ECCV2020](https://arxiv.org/pdf/2007.10982.pdf)] Shape and Viewpoint without Keypoints
- [[Arxiv](https://arxiv.org/pdf/2007.10300.pdf)] Object-Centric Multi-View Aggregation
- [[ECCV2020](https://arxiv.org/pdf/2007.10453.pdf)] Points2Surf Learning Implicit Surfaces from Point Clouds
- [[NeurIPS2020](https://arxiv.org/pdf/2007.10973.pdf)] Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows [[Project](https://kunalmgupta.github.io/projects/NeuralMeshflow.html)]
- [[Arxiv](https://arxiv.org/pdf/2006.12250.pdf)] Pix2Vox++: Multi-scale Context-aware 3D Object Reconstruction from Single and Multiple Images
- [[Arxiv](https://arxiv.org/pdf/2006.13240.pdf)] Neural Non-Rigid Tracking
- [[NeurIPS2020](https://arxiv.org/pdf/2006.03997.pdf)] MeshSDF: Differentiable Iso-Surface Extraction
- [[Arxiv](https://arxiv.org/pdf/2006.07752.pdf)] 3D Reconstruction of Novel Object Shapes from Single Images
- [[NeurIPS2020](https://arxiv.org/pdf/2006.07982.pdf)] ShapeFlow: Learnable Deformations Among 3D Shapes [[pytorch](https://github.com/maxjiang93/ShapeFlow)]
- [[Arxiv](https://arxiv.org/pdf/2006.09694.pdf)] 3D Shape Reconstruction from Free-Hand Sketches
- [[Arxiv](https://arxiv.org/pdf/2003.08934.pdf)] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
- [[Arxiv](https://arxiv.org/pdf/2003.04618.pdf)] Convolutional Occupancy Networks
- [[Siggraph2020](https://arxiv.org/pdf/2005.11084.pdf)] Point2Mesh: A Self-Prior for Deformable Meshes
- [[Arxiv](https://arxiv.org/pdf/2005.02138.pdf)] PointTriNet: Learned Triangulation of 3D Point
- [[Arxiv](https://arxiv.org/pdf/2005.04623.pdf)] A Simple and Scalable Shape Representation for 3D Reconstruction
- [[Siggraph2020](https://arxiv.org/pdf/2005.03372.pdf)] Vid2Curve: Simultaneously Camera Motion Estimation and Thin Structure Reconstruction from an RGB Video
- [[CVPR2020](https://arxiv.org/pdf/2005.01939.pdf)] From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks [[tensorflow](https://github.com/val-iisc/ssl_3d_recon)]
- [[CVPR2020](https://arxiv.org/pdf/2004.10904.pdf)] Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes [[github](https://github.com/lzqsd/TransparentShapeReconstruction)]
- [[Arxiv](https://arxiv.org/pdf/2002.10880.pdf)] PolyGen: An Autoregressive Generative Model of 3D Meshes
- [[Arxiv](https://arxiv.org/pdf/2004.07414.pdf)] Combinatorial 3D Shape Generation via Sequential Assembly
- [[Arxiv](https://arxiv.org/pdf/2004.06302.pdf)] Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors
- [[Arxiv](https://arxiv.org/pdf/2004.04485.pdf)] Neural Object Descriptors for Multi-View Shape Reconstruction
- [[CVPR2020](https://arxiv.org/pdf/2003.14034.pdf)] SPARE3D: A Dataset for SPAtial REasoning on Three-View Line Drawings [[pytorch](https://github.com/ai4ce/SPARE3D)]
- [[Arxiv](https://arxiv.org/pdf/2003.12397.pdf)] Modeling 3D Shapes by Reinforcement Learning
- [[ECCV2020](https://arxiv.org/pdf/2003.12181.pdf)] ParSeNet: A Parametric Surface Fitting Network for 3D Point Clouds [[pytorch](https://github.com/Hippogriff/parsenet-codebase)]
- [[Arxiv](https://arxiv.org/pdf/2003.10016.pdf)] Self-Supervised 2D Image to 3D Shape Translation with Disentangled Representations
- [[Arxiv](https://arxiv.org/pdf/2003.09852.pdf)] Universal Differentiable Renderer for Implicit Neural Representations
- [[Arxiv](https://arxiv.org/pdf/2003.09754.pdf)] Learning 3D Part Assembly from a Single Image
- [[Arxiv](https://arxiv.org/pdf/2003.08593.pdf)] Curriculum DeepSDF
- [[Arxiv](https://arxiv.org/pdf/2003.08624.pdf)] PT2PC: Learning to Generate 3D Point Cloud Shapes from Part Tree Conditions
- [[Arxiv](https://arxiv.org/pdf/2003.06473.pdf)] Self-supervised Single-view 3D Reconstruction via Semantic Consistency
- [[Arxiv](https://arxiv.org/pdf/2003.03711.pdf)] Meta3D: Single-View 3D Object Reconstruction from Shape Priors in Memory
- [[Arxiv](https://arxiv.org/pdf/2003.03551.pdf)] STD-Net: Structure-preserving and Topology-adaptive Deformation Network for 3D Reconstruction from a Single Image
- [[Arxiv](https://arxiv.org/pdf/2001.07884.pdf)] Curvature Regularized Surface Reconstruction from Point Cloud
- [[Arxiv](https://arxiv.org/pdf/2003.00802.pdf)] Hypernetwork approach to generating point clouds
- [[Arxiv](https://arxiv.org/pdf/2002.12674.pdf)] Inverse Graphics GAN: Learning to Generate 3D Shapes from Unstructured 2D Data
- [[Arxiv](https://arxiv.org/pdf/2001.01744.pdf)] Meshlet Priors for 3D Mesh Reconstruction
- [[Arxiv](https://arxiv.org/pdf/1912.10589.pdf)] Front2Back: Single View 3D Shape Reconstruction via Front to Back Prediction
- [[Arxiv](https://arxiv.org/pdf/1912.07109.pdf)] SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization
- [[CVPR2019](https://arxiv.org/pdf/1812.03828.pdf)] Occupancy Networks: Learning 3D Reconstruction in Function Space [[pytorch](https://github.com/autonomousvision/occupancy_networks)] :fire::star:
- [[NeurIPS2019](https://arxiv.org/pdf/1905.10711.pdf)] DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction [[tensorflow](https://github.com/laughtervv/DISN)]
- [[NeurIPS2019](https://arxiv.org/pdf/1905.10711.pdf)] Learning to Infer Implicit Surfaces without 3D Supervision
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_A_Skeleton-Bridged_Deep_Learning_Approach_for_Generating_Meshes_of_Complex_CVPR_2019_paper.pdf)] A Skeleton-bridged Deep Learning Approach for Generating Meshes of Complex Topologies from Single RGB Images [[pytorch & tensorflow](https://github.com/tangjiapeng/SkeletonBridgeRecon)]
- [[Arxiv](https://arxiv.org/pdf/1901.06802.pdf)] Deep Level Sets: Implicit Surface Representations for 3D Shape Inference
- [[CVPR2019](https://arxiv.org/pdf/1812.02822.pdf)] Learning Implicit Fields for Generative Shape Modeling [[tensorflow](https://github.com/czq142857/implicit-decoder)] :fire:
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.pdf)] Point-based Multi-view Stereo Network [[pytorch](https://github.com/callmeray/PointMVSNet)] :star:
- [[Arxiv](https://arxiv.org/pdf/1911.07401.pdf)] TSRNet: Scalable 3D Surface Reconstruction Network for Point Clouds using Tangent Convolution
- [[Arxiv](https://arxiv.org/ftp/arxiv/papers/1911/1911.09204.pdf)] DR-KFD: A Differentiable Visual Metric for 3D Shape Reconstruction
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nguyen_GraphX-Convolution_for_Point_Cloud_Deformation_in_2D-to-3D_Conversion_ICCV_2019_paper.pdf)] GraphX-Convolution for Point Cloud Deformation in 2D-to-3D Conversion
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wen_Pixel2Mesh_Multi-View_3D_Mesh_Generation_via_Deformation_ICCV_2019_paper.pdf)] Pixel2Mesh++: Multi-View 3D Mesh Generation via Deformation [[pytorch](https://github.com/walsvid/Pixel2MeshPlusPlus)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wallace_Few-Shot_Generalization_for_Single-Image_3D_Reconstruction_via_Priors_ICCV_2019_paper.pdf)] Few-Shot Generalization for Single-Image 3D Reconstruction via Priors
- [[ICCV2019](https://arxiv.org/pdf/1909.00321.pdf)] Deep Mesh Reconstruction from Single RGB Images via Topology Modification Networks
- [[AAAI2018](https://arxiv.org/pdf/1706.07036.pdf)] Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction [[tensorflow](https://github.com/chenhsuanlin/3D-point-cloud-generation)] :star::fire:
- [[NeurIPS2017](https://papers.nips.cc/paper/6657-marrnet-3d-shape-reconstruction-via-25d-sketches.pdf)] MarrNet: 3D Shape Reconstruction via 2.5D Sketches [[torch](https://github.com/jiajunwu/marrnet)]:star::fire:





---
## 3D Scene Understanding
- [[Arxiv](https://arxiv.org/pdf/2102.08945.pdf)] Weakly Supervised Learning of Rigid 3D Scene Flow
- [[ICLR2021](https://arxiv.org/pdf/2102.07764.pdf)] End-to-End Egospheric Spatial Memory
- [[Arxiv](https://arxiv.org/pdf/2102.03939.pdf)] Single-Shot Cuboids: Geodesics-based End-to-end Manhattan Aligned Layout
Estimation from Spherical Panoramas [[Project](https://vcl3d.github.io/SingleShotCuboids/)]
- [[Arxiv](https://arxiv.org/pdf/2101.07891.pdf)] A modular vision language navigation and manipulation framework for long horizon compositional tasks in indoor environment
- [[Arxiv](https://arxiv.org/pdf/2101.07462.pdf)] Deep Reinforcement Learning for Producing Furniture Layout in Indoor Scenes
- [[Arxiv](https://arxiv.org/pdf/2101.02692.pdf)] Where2Act: From Pixels to Actions for Articulated 3D Objects [[Project](https://cs.stanford.edu/~kaichun/where2act/)]

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/1712.05474.pdf)] AI2-THOR: An Interactive 3D Environment for Visual AI [[Project](https://ai2thor.allenai.org/)]
- [[Arxiv](https://arxiv.org/pdf/2012.15470.pdf)] Audio-Visual Floorplan Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2012.00987.pdf)] PV-RAFT: Point-Voxel Correlation Fields for Scene Flow Estimation of Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2012.00726.pdf)] RAFT-3D: Scene Flow using Rigid-Motion Embeddings
- [[Arxiv](https://arxiv.org/abs/2012.03998)] GenScan: A Generative Method for Populating Parametric 3D Scan Datasets
- [[Arxiv](https://arxiv.org/pdf/2012.06547.pdf)] LayoutGMN: Neural Graph Matching for Structural Layout Similarity
- [[Arxiv](https://arxiv.org/pdf/2012.08197.pdf)] Seeing Behind Objects for 3D Multi-Object Tracking in RGB-D Sequences
- [[Arxiv](https://arxiv.org/pdf/2012.13089.pdf)] P4Contrast: Contrastive Learning with Pairs of Point-Pixel Pairs for RGB-D Scene Understanding
- [[Arxiv](https://arxiv.org/pdf/2012.12395.pdf)] Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion
Forecasting with a Single Convolutional Net
- [[Arxiv](https://arxiv.org/pdf/2011.04122.pdf)] Localising In Complex Scenes Using Balanced Adversarial Adaptation
- [[Arxiv](https://arxiv.org/pdf/2011.06961.pdf)] Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis
- [[NeurIPS2020](https://arxiv.org/pdf/2011.10007.pdf)] Multi-Plane Program Induction with 3D Box Priors [[Project](http://bpi.csail.mit.edu/)]
- [[Arxiv](https://arxiv.org/pdf/2011.11498.pdf)] HoHoNet: 360 Indoor Holistic Understanding with Latent Horizontal Features
- [[Arxiv](https://arxiv.org/pdf/2012.09165.pdf)] Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts
- [[Arxiv](https://arxiv.org/pdf/2011.13417.pdf)] Generative Layout Modeling using Constraint Graphs
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/76dc611d6ebaafc66cc0879c71b5db5c-Paper.pdf)] Rel3D: A Minimally Contrastive Benchmark for Grounding Spatial Relations in 3D [[pytorch](https://github.com/princeton-vl/Rel3D)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf)] Learning Affordance Landscapes for Interaction Exploration in 3D Environments [[Project](http://vision.cs.utexas.edu/projects/interaction-exploration/)]
- [[NeurIPS2020W](https://arxiv.org/abs/2010.14543)] Unsupervised Domain Adaptation for Visual Navigation
- [[Arxiv](https://arxiv.org/pdf/2009.05429.pdf)] Embodied Visual Navigation with Automatic Curriculum Learningin Real Environments
- [[Arxiv](https://arxiv.org/abs/2009.02857)] 3D Room Layout Estimation Beyond the Manhattan World Assumption
- [[Arxiv](https://arxiv.org/pdf/2008.10631.pdf)] OpenBot: Turning Smartphones into Robots [[Project](https://www.openbot.org/)]
- [[Arxiv](https://arxiv.org/abs/2008.09622)] Audio-Visual Waypoints for Navigation
- [[Arxiv](https://arxiv.org/pdf/2008.09241.pdf)] Learning Affordance Landscapes for Interaction Exploration in 3D Environments [[Project](http://vision.cs.utexas.edu/projects/interaction-exploration/)]
- [[ECCV2020](https://arxiv.org/pdf/2008.09285.pdf)] Occupancy Anticipation for Efficient Exploration and Navigation [[Project](http://vision.cs.utexas.edu/projects/occupancy_anticipation/)]
- [[Arxiv](https://arxiv.org/pdf/2008.07817.pdf)] Retargetable AR: Context-aware Augmented Reality in Indoor Scenes based on 3D Scene Graph
- [[Arxiv](https://arxiv.org/abs/2008.05570)] Generating Person-Scene Interactions in 3D Scenes
- [[Arxiv](https://arxiv.org/abs/2008.06286)] GeoLayout: Geometry Driven Room Layout Estimation Based on Depth Maps of Planes
- [[ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf)] ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes
- [[Arxiv](https://arxiv.org/pdf/2008.01323.pdf)] Structural Plan of Indoor Scenes with Personalized Preferences
- [[Arxiv](https://arxiv.org/abs/2008.03286)] HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures [[Project](https://people.eecs.berkeley.edu/~zyc/holicity/)]
- [[CVPR2020](https://arxiv.org/abs/2007.11744)] End-to-End Optimization of Scene Layout [[Project](http://3dsln.csail.mit.edu/)]
- [[Arxiv](https://arxiv.org/pdf/2005.02153.pdf)] Improving Target-driven Visual Navigation with Attention on 3D Spatial Relationships
- [[CVPR2020](https://arxiv.org/pdf/2004.03967.pdf)] Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions
- [[Arxiv](https://arxiv.org/pdf/2003.13516.pdf)] LayoutMP3D: Layout Annotation of Matterport3D
- [[CVPR2020](https://arxiv.org/pdf/2003.08981.pdf)] Local Implicit Grid Representations for 3D Scenes
- [[Arxiv](https://arxiv.org/pdf/2003.07356.pdf)] Scan2Plan: Efficient Floorplan Generation from 3D Scans of Indoor Scenes
- [[CVPR2020](https://arxiv.org/pdf/1911.11236.pdf)] RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds [[tensorflow](https://github.com/QingyongHu/RandLA-Net)] :fire:
- [[CVPR2020](https://arxiv.org/pdf/2003.00397.pdf)] Intelligent Home 3D: Automatic 3D-House Design from Linguistic Descriptions Only
- [[ICRA2020](https://arxiv.org/pdf/2003.00535.pdf)] 3DCFS: Fast and Robust Joint 3D Semantic-Instance Segmentation via Coupled Feature Selection
- [[Arxiv](https://arxiv.org/pdf/2002.12819.pdf)] Indoor Scene Recognition in 3D
- [[Journal](https://arxiv.org/pdf/2002.08988.pdf)] Dark, Beyond Deep: A Paradigm Shift to Cognitive AI with Humanlike Common Sense
- [[Arxiv](https://arxiv.org/pdf/2002.08988.pdf)] BlockGAN Learning 3D Object-aware Scene Representations from Unlabelled Images
- [[Arxiv](https://arxiv.org/abs/2002.06289)] 3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans [[Project](https://github.com/MIT-SPARK/Kimera)] Related: [[Arxiv](https://arxiv.org/abs/1910.02490)] [[Arxiv](https://arxiv.org/abs/2101.06894)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Mustafa_U4D_Unsupervised_4D_Dynamic_Scene_Understanding_ICCV_2019_paper.pdf)] U4D: Unsupervised 4D Dynamic Scene Understanding
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xian_UprightNet_Geometry-Aware_Camera_Orientation_Estimation_From_Single_Images_ICCV_2019_paper.pdf)] UprightNet: Geometry-Aware Camera Orientation Estimation from Single Images
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Savva_Habitat_A_Platform_for_Embodied_AI_Research_ICCV_2019_paper.pdf)] Habitat: A Platform for Embodied AI Research [[habitat-api](https://github.com/facebookresearch/habitat-api)] [[habitat-sim](https://github.com/facebookresearch/habitat-sim)] :star:
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.pdf)] SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences [[project page](http://semantic-kitti.org/)] :star:
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sengupta_Neural_Inverse_Rendering_of_an_Indoor_Scene_From_a_Single_ICCV_2019_paper.pdf)] Neural Inverse Rendering of an Indoor Scene From a Single Image
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_SceneGraphNet_Neural_Message_Passing_for_3D_Indoor_Scene_Augmentation_ICCV_2019_paper.pdf)] SceneGraphNet: Neural Message Passing for 3D Indoor Scene Augmentation [[pytorch](https://github.com/yzhou359/3DIndoor-SceneGraphNet)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf)] RIO: 3D Object Instance Re-Localization in Changing Indoor Environments [[dataset](https://github.com/WaldJohannaU/3RScan)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_CamNet_Coarse-to-Fine_Retrieval_for_Camera_Re-Localization_ICCV_2019_paper.pdf)] CamNet: Coarse-to-Fine Retrieval for Camera Re-Localization
- [[ICCV2019](https://arxiv.org/pdf/1907.09905.pdf)] U4D: Unsupervised 4D Dynamic Scene Understanding
- [[NeurIPS2018](https://papers.nips.cc/paper/7444-learning-to-exploit-stability-for-3d-scene-parsing.pdf)] Learning to Exploit Stability for 3D Scene Parsing





---
## 3D Scene Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2102.13090.pdf)] IBRNet: Learning Multi-View Image-Based Rendering [[Project](https://ibrnet.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2102.07064)] NeRF--: Neural Radiance Fields Without Known Camera Parameters [[Project](http://nerfmm.active.vision/)]
- [[Arxiv](https://arxiv.org/pdf/2012.02190.pdf)] STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in Motion with Neural Rendering [[Project](https://wentaoyuan.github.io/star/)]

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.02190.pdf)] pixelNeRF: Neural Radiance Fields from One or Few Images [[Project](https://alexyu.net/pixelnerf/)]
- [[Arxiv](https://arxiv.org/pdf/2012.03927.pdf)] NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis [[Project](https://people.eecs.berkeley.edu/~pratul/nerv/)]
- [[Arxiv](https://arxiv.org/pdf/2012.05360.pdf)] MO-LTR: Multiple Object Localization, Tracking and Reconstruction from Monocular RGB Videos
- [[Arxiv](https://arxiv.org/pdf/2012.05551.pdf)] DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors
- [[Arxiv](https://arxiv.org/pdf/2012.09790.pdf)] Neural Radiance Flow for 4D View Synthesis and Video Processing [[Project](https://yilundu.github.io/nerflow/)]
- [[3DV2020](https://arxiv.org/pdf/2011.00320.pdf)] Scene Flow from Point Clouds with or without Learning
- [[Arxiv](https://arxiv.org/abs/2011.07233)] Stable View Synthesis
- [[Arxiv](https://arxiv.org/pdf/2011.10379.pdf)] Neural Scene Graphs for Dynamic Scenes
- [[3DV2020](https://arxiv.org/pdf/2011.10359.pdf)] RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty [[pytorch](https://github.com/facebookresearch/RidgeSfM)]
- [[Arxiv](https://arxiv.org/pdf/2011.10147.pdf)] FlowStep3D: Model Unrolling for Self-Supervised Scene Flow Estimation
- [[Arxiv](https://arxiv.org/pdf/2011.10812.pdf)] MoNet: Motion-based Point Cloud Prediction Network
- [[Arxiv](https://arxiv.org/pdf/2011.11814.pdf)] MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments from a Single Moving Camera
- [[Arxiv](https://arxiv.org/pdf/2011.11986.pdf)] Efficient Initial Pose-graph Generation for Global SfM
- [[Arxiv](https://arxiv.org/abs/2011.13084)] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes [[Project](http://www.cs.cornell.edu/~zl548/NSFF/)]
- [[Arxiv](https://arxiv.org/abs/2011.12948)] Deformable Neural Radiance Fields [[Project](https://nerfies.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2011.12490.pdf)] DeRF: Decomposed Radiance Fields
- [[Arxiv](https://arxiv.org/pdf/2011.14398.pdf)] RGBD-Net: Predicting color and depth images for novel views synthesis
- [[Arxiv](https://arxiv.org/pdf/2012.04512.pdf)] SSCNav: Confidence-Aware Semantic Scene Completion for Visual Semantic Navigation [[Project](https://sscnav.cs.columbia.edu/)]
- [[Arxiv](https://arxiv.org/pdf/2012.11575.pdf)] From Points to Multi-Object 3D Reconstruction
- [[Arxiv](https://worldsheet.github.io/resources/worldsheet.pdf)] Worldsheet: Wrapping the World in a 3D Sheet
for View Synthesis from a Single Image [[Project](https://worldsheet.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2012.09793.pdf)] SceneFormer: Indoor Scene Generation with Transformers
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/b4b758962f17808746e9bb832a6fa4b8-Paper.pdf)] Neural Sparse Voxel Fields [[Project](https://lingjie0206.github.io/papers/NSVF/)]
- [[Arxiv](https://arxiv.org/pdf/2012.02094.pdf?fbclid=IwAR03XwEdhXUl2lsLr20dOnFEsnthPBbdVi9VHDni6CYnhH9glzGaooU-DHM)] Towards Part-Based Understanding of RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/2011.05813.pdf)] Dynamic Plane Convolutional Occupancy Networks
- [[NeurIPS2020](https://arxiv.org/pdf/2010.13938.pdf)] Neural Unsigned Distance Fields for Implicit
Function Learning [[Project](http://virtualhumans.mpi-inf.mpg.de/ndf/)]
- [[Arxiv](https://arxiv.org/pdf/2010.01549.pdf)] Holistic static and animated 3D scene generation from diverse text descriptions [[pytorch](https://github.com/oaishi/3DScene_from_text)]
- [[Arxiv](https://arxiv.org/pdf/2010.04030.pdf)] Semi-Supervised Learning of Multi-Object 3D Scene Representations
- [[ECCV2020](https://arxiv.org/pdf/2007.11965.pdf)] CAD-Deform: Deformable Fitting of CAD Models to 3D Scans
- [[ECCV2020](https://arxiv.org/pdf/2007.13034.pdf)] Mask2CAD: 3D Shape Prediction by Learning to Segment and Retrieve
- [[ECCV2020](https://arxiv.org/pdf/2007.11431.pdf)] Learnable Cost Volume Using the Cayley Representation
- [[ECCV2020](https://arxiv.org/pdf/2007.06853.pdf)] Topology-Change-Aware Volumetric Fusion for Dynamic Scene Reconstruction
- [[ECCV2020](https://arxiv.org/pdf/2003.04618.pdf)] Convolutional Occupancy Networks
- [[CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_MARMVS_Matching_Ambiguity_Reduced_Multiple_View_Stereo_for_Efficient_Large_CVPR_2020_paper.pdf)] MARMVS: Matching Ambiguity Reduced Multiple View Stereo for Efficient Large Scale Scene Reconstruction
- [[ECCV2020](https://arxiv.org/pdf/2004.12989.pdf)] CoReNet: Coherent 3D scene reconstruction from a single RGB image
- [[CVPR2020](https://arxiv.org/pdf/2004.01170.pdf)] DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes
- [[ECCV2020](https://arxiv.org/pdf/2003.12622.pdf)] SceneCAD: Predicting Object Alignments and Layouts in RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/2003.11076.pdf)] Removing Dynamic Objects for Static Scene Reconstruction using Light Fields
- [[Arxiv](https://arxiv.org/pdf/2003.10432.pdf)] Atlas: End-to-End 3D Scene Reconstruction from Posed Images
- [[Arxiv](https://arxiv.org/pdf/2003.07356.pdf)] Scan2Plan: Efficient Floorplan Generation from 3D Scans of Indoor Scenes
- [[Arxiv](https://arxiv.org/pdf/2001.07058.pdf)] Plane Pair Matching for Efficient 3D View Registration
- [[CVPR2020](https://arxiv.org/pdf/2002.12212.pdf)] Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image [[pytorch](https://github.com/yinyunie/Total3DUnderstanding)]
- [[Arxiv](https://arxiv.org/pdf/2001.05422.pdf)] Indoor Layout Estimation by 2D LiDAR and Camera Fusion
- [[Arxiv](https://arxiv.org/pdf/2001.02149.pdf)] General 3D Room Layout from a Single View by Render-and-Compare
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Learning_to_Reconstruct_3D_Manhattan_Wireframes_From_a_Single_Image_ICCV_2019_paper.pdf)] Learning to Reconstruct 3D Manhattan Wireframes from a Single Image
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_PlaneRCNN_3D_Plane_Detection_and_Reconstruction_From_a_Single_Image_CVPR_2019_paper.pdf)] PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image [[pytorch](https://github.com/NVlabs/planercnn)]:fire:
- [[ICCV2019](https://arxiv.org/pdf/1902.06729.pdf)] 3D Scene Reconstruction with Multi-layer Depth and Epipolar Transformers
- [[ICCV Workshop2019](http://openaccess.thecvf.com/content_ICCVW_2019/papers/3DRW/Li_Silhouette-Assisted_3D_Object_Instance_Reconstruction_from_a_Cluttered_Scene_ICCVW_2019_paper.pdf)] Silhouette-Assisted 3D Object Instance Reconstruction from a Cluttered Scene
- [[ICCV2019](https://arxiv.org/pdf/1906.02729.pdf)] 3D-RelNet: Joint Object and Relation Network for 3D prediction [[pytorch](https://github.com/nileshkulkarni/relative3d)]
- [[3DV2019](https://arxiv.org/pdf/1907.00939.pdf)] Pano Popups: Indoor 3D Reconstruction with a Plane-Aware Network
- [[CVPR2018](https://arxiv.org/pdf/1712.01812.pdf)] Factoring Shape, Pose, and Layout from the 2D Image of a 3D Scene [[pytorch](https://github.com/shubhtuls/factored3d)]
- [[IROS2017](https://www.microsoft.com/en-us/research/uploads/prod/2019/09/MSrivathsan2017IROS.pdf)] Indoor Scan2BIM: Building Information Models of House Interiors
- [[CVPR2017](https://arxiv.org/pdf/1603.08182.pdf)] 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions [[github](https://github.com/andyzeng/3dmatch-toolbox)]





---
## About Human Body
- [[Arxiv](https://arxiv.org/pdf/2011.15079.pdf)] Forecasting Characteristic 3D Poses of Human Actions
- [[Arxiv](https://arxiv.org/pdf/2102.07343.pdf)] Capturing Detailed Deformations of Moving Human Bodies
- [[Arxiv](https://arxiv.org/abs/2102.06199)] A-NeRF: Surface-free Human 3D Pose Refinement via Neural Rendering
 [[Project](https://lemonatsu.github.io/ANeRF-Surface-free-Pose-Refinement/)]
- [[Arxiv](https://arxiv.org/abs/2101.08779v1)] Learn to Dance with AIST++: Music Conditioned 3D Dance Generation [[Project](https://google.github.io/aistplusplus_dataset/)]
- [[Arxiv](https://arxiv.org/pdf/2101.06571.pdf)] S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling
- [[Arxiv](https://arxiv.org/pdf/2101.02471.pdf)] PandaNet : Anchor-Based Single-Shot Multi-Person 3D Pose Estimation
- [[Arxiv](https://arxiv.org/pdf/2012.15838.pdf)] Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans [[Project](https://zju3dv.github.io/neuralbody/)]
- [[Arxiv](https://arxiv.org/abs/2012.14739)] Chasing the Tail in Monocular 3D Human Reconstruction with Prototype Memory
- [[3DV2020](https://arxiv.org/pdf/2008.05570.pdf)] PLACE: Proximity Learning of Articulation and Contact in 3D Environments [[Project](https://sanweiliti.github.io/PLACE/PLACE.html)]
- [[ICCV2019](https://arxiv.org/abs/1908.06963)] Resolving 3D Human Pose Ambiguities with 3D Scene Constraints [[Project](https://prox.is.tue.mpg.de/)]

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.05522.pdf)] Synthesizing Long-Term 3D Human Motion and Interaction in 3D Scenes [[Project](https://jiashunwang.github.io/Long-term-Motion-in-3D-Scenes/)]
- [[Arxiv](https://arxiv.org/pdf/2012.09760.pdf)] End-to-End Human Pose and Mesh Reconstruction with Transformers
- [[Arxiv](https://arxiv.org/pdf/2012.09843.pdf)] Human Mesh Recovery from Multiple Shots [[Project](https://geopavlakos.github.io/multishot/)]
- [[NeurIPS2020](https://arxiv.org/pdf/2011.00980.pdf)] 3D Multi-bodies: Fitting Sets of Plausible 3D Human Models to Ambiguous Image Data [[Project](https://sites.google.com/view/3dmb/home)]
- [[Arxiv](https://arxiv.org/pdf/2012.01591.pdf)] Holistic 3D Human and Scene Mesh Estimation from Single View Images
- [[Arxiv](https://arxiv.org/pdf/2011.08627.pdf)] Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video
- [[Arxiv](https://arxiv.org/pdf/2011.11534.pdf)] Pose2Pose: 3D Positional Pose-Guided 3D Rotational Pose Prediction
for Expressive 3D Human Pose and Mesh Estimation
- [[Arxiv](https://arxiv.org/pdf/2011.11232.pdf)] NeuralAnnot: Neural Annotator for in-the-wild Expressive 3D Human Pose and Mesh Training Sets
- [[Arxiv](https://arxiv.org/pdf/2011.13341.pdf)] 4D Human Body Capture from Egocentric Video via 3D Scene Grounding [[Project](https://aptx4869lm.github.io/4DEgocentricBodyCapture/)]
- [[Arxiv](https://posa.is.tue.mpg.de/)] Populating 3D Scenes by Learning Human-Scene Interaction [[Project](https://posa.is.tue.mpg.de/)]
- [[ECCV2020](https://arxiv.org/pdf/2007.03672.pdf)] Long-term Human Motion Prediction with Scene Context [[Project](https://people.eecs.berkeley.edu/~zhecao/hmp/index.html)]
- [[Arxiv](https://arxiv.org/abs/2012.12884)] Vid2Actor: Free-viewpoint Animatable Person Synthesis from Video in the Wild [[Project](https://grail.cs.washington.edu/projects/vid2actor/)]
- [[Arxiv](https://arxiv.org/pdf/2012.12890.pdf)] ANR: Articulated Neural Rendering for Virtual Avatars
- [[Arxiv](https://arxiv.org/pdf/1912.02923.pdf)] Generating 3D People in Scenes without People [[Project](https://github.com/yz-cnsdqz/PSI-release)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Holistic_Scene_Understanding_Single-View_3D_Holistic_Scene_Parsing_and_Human_ICCV_2019_paper.pdf)] Holistic++ Scene Understanding: Single-view 3D Holistic Scene Parsing and Human Pose Estimation with Human-Object Interaction and Physical Commonsense
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Putting_Humans_in_a_Scene_Learning_Affordance_in_3D_Indoor_CVPR_2019_paper.pdf)] Putting Humans in a Scene: Learning Affordance in 3D Indoor Environments
- [[TOG2016](https://graphics.stanford.edu/projects/pigraphs/pigraphs.pdf)] Pigraphs: learning interaction
snapshots from observations [[Project](https://graphics.stanford.edu/projects/pigraphs/)]





---
## General Methods
- [[Arxiv](https://arxiv.org/abs/2102.06171)] High-Performance Large-Scale Image Recognition Without Normalization
- [[Arxiv](https://arxiv.org/pdf/2102.04776.pdf)] Generative Models as Distributions of Functions
- [[Arxiv](https://arxiv.org/pdf/2102.04014.pdf)] Point-set Distances for Learning Representations of 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2102.02896.pdf)] Compressed Object Detection
- [[Arxiv](https://arxiv.org/pdf/2102.00084.pdf)] A linearized framework and a new benchmark for model selection for fine-tuning
- [[Arxiv](https://arxiv.org/pdf/2101.07832.pdf)] The Devils in the Point Clouds: Studying the Robustness of Point Cloud Convolutions
- [[Arxiv](https://arxiv.org/pdf/2101.02691.pdf)] Self-Supervised Pretraining of 3D Features on any Point-Cloud [[pytorch](https://github.com/facebookresearch/DepthContrast)]
- [[3DV2020](https://arxiv.org/pdf/2101.00483.pdf)] Learning Rotation-Invariant Representations of Point Clouds Using Aligned Edge Convolutional Neural Networks

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.00888.pdf)] Diffusion is All You Need for Learning on Surfaces
- [[Arxiv](https://arxiv.org/pdf/2012.04439.pdf)] SPU-Net: Self-Supervised Point Cloud Upsampling by Coarse-to-Fine Reconstruction with Self-Projection Optimization
- [[3DV2020](https://arxiv.org/pdf/2012.04048.pdf)] Rotation-Invariant Point Convolution With Multiple Equivariant Alignments
- [[Arxiv](https://arxiv.org/pdf/2012.06257.pdf)] One Point is All You Need: Directional Attention Point for Feature Learning
- [[Arxiv](https://arxiv.org/pdf/2012.09688.pdf)] PCT: Point Cloud Transformer
- [[Arxiv](https://arxiv.org/pdf/2012.13118.pdf)] Hausdorff Point Convolution with Geometric Priors
- [[Arxiv](https://arxiv.org/pdf/2011.00923.pdf)] MARNet: Multi-Abstraction Refinement Network for 3D Point Cloud Analysis [[Github](https://github.com/ruc98/MARNet)]
- [[Arxiv](https://arxiv.org/pdf/2011.00931.pdf)] Point Transformer
- [[Arxiv](https://arxiv.org/pdf/2011.14289.pdf)] Learning geometry-image representation for 3D point cloud generation
- [[Arxiv](https://arxiv.org/pdf/2011.14285.pdf)] Deeper or Wider Networks of Point Clouds with Self-attention?
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/0a656cc19f3f5b41530182a9e03982a4-Paper.pdf)] Primal-Dual Mesh Convolutional Neural Networks [[pytorch](https://github.com/MIT-SPARK/PD-MeshNet)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/a3f390d88e4c41f2747bfa2f1b5f87db-Paper.pdf)] Rational neural networks [[tensorflow](https://github.com/NBoulle/RationalNets)]
- [[NeurIPS2020](https://arxiv.org/abs/2008.02676)] Exchangeable Neural ODE for Set Modeling [[Project](https://github.com/lupalab/ExNODE)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)] SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks [[Project](https://fabianfuchsml.github.io/se3transformer/)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf)] NVAE: A Deep Hierarchical Variational Autoencoder [[pytorch](https://github.com/NVlabs/NVAE)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf)] Implicit Graph Neural Networks [[pytorch](https://github.com/SwiftieH/IGNN)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/ac10ff1941c540cd87c107330996f4f6-Paper.pdf)] The Autoencoding Variational Autoencoder [[pytorch](https://github.com/snap-stanford/graphgym)]
- [[Arxiv](https://arxiv.org/pdf/2010.07215.pdf)] PointManifold: Using Manifold Learning for Point Cloud Classification
- [[Arxiv](https://arxiv.org/pdf/2010.15831.pdf)] RelationNet++: Bridging Visual Representations for
Object Detection via Transformer Decoder
- [[Arxiv](https://arxiv.org/pdf/2010.01089.pdf)] Pre-Training by Completing Point Clouds [[pytorch](https://github.com/hansen7/OcCo)]
- [[NeurIPS2020](https://arxiv.org/pdf/2010.03318.pdf)] Rotation-Invariant Local-to-Global Representation Learning for 3D Point Cloud
- [[Arxiv](https://arxiv.org/abs/2010.05272)] IF-Defense: 3D Adversarial Point Cloud Defense via Implicit Function based Restoration [[pytorch](https://github.com/Wuziyi616/IF-Defense)]
- [[Arxiv](https://arxiv.org/abs/2009.02918)] DV-ConvNet: Fully Convolutional Deep Learning on Point Clouds with Dynamic Voxelization and 3D Group Convolution
- [[Arxiv](https://arxiv.org/pdf/2009.01427.pdf)] Spatial Transformer Point Convolution
- [[Arxiv](https://arxiv.org/pdf/2008.12066.pdf)] Minimal Adversarial Examples for Deep Learning on 3D Point Clouds
- [[BMVC2020](https://arxiv.org/abs/2008.05981)] Black Magic in Deep Learning: How Human Skill Impacts Network Training
- [[ECCV2020](https://arxiv.org/pdf/2008.06374.pdf)] PointMixup: Augmentation for Point Clouds [[Code](https://github.com/yunlu-chen/PointMixup/)]
- [[ECCV2020](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660290.pdf)] DR-KFS: A Differentiable Visual Similarity Metric for 3D Shape Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2008.01068.pdf)] Unsupervised 3D Learning for Shape Analysis via Multiresolution Instance Discrimination
- [[Arxiv](https://arxiv.org/pdf/2008.02986.pdf)] Global Context Aware Convolutions for 3D Point Cloud Understanding
- [[ECCV2020](https://arxiv.org/pdf/2008.00892.pdf)] Shape Adaptor: A Learnable Resizing Module [[pytorch](https://github.com/lorenmt/shape-adaptor)]
- [[ACMMM2020](https://arxiv.org/pdf/2007.13551.pdf)] Differentiable Manifold Reconstruction for Point Cloud Denoising [[pytorch](https://github.com/luost26/DMRDenoise)]
- [[ECCV2020](https://arxiv.org/pdf/2007.10170.pdf)] Discrete Point Flow Networks for Efficient Point Cloud Generation
- [[Siggraph2020](https://www.dgp.toronto.edu/projects/neural-subdivision/)] Neural Subdivision
- [[Arxiv](https://arxiv.org/pdf/2007.10985.pdf)] PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding
- [[Arxiv](https://arxiv.org/pdf/2007.08501.pdf)] Accelerating 3D Deep Learning with PyTorch3D
- [[Arxiv](https://arxiv.org/pdf/2007.08349.pdf)] Natural Graph Networks
- [[ECCV2020](https://arxiv.org/pdf/2007.05361.pdf)] Progressive Point Cloud Deconvolution Generation Network [[github](https://github.com/fpthink/PDGN)]
- [[Arxiv](https://arxiv.org/pdf/2007.04537.pdf)] Point Set Voting for Partial Point Cloud Analysis
- [[Arxiv](https://arxiv.org/pdf/2007.04525.pdf)] PointMask: Towards Interpretable and Bias-Resilient Point Cloud Processing
- [[Arxiv](https://arxiv.org/pdf/2006.04325.pdf)] Fully Convolutional Mesh Autoencoder using Efficient Spatially Varying Kernels
- [[Arxiv](https://arxiv.org/pdf/2007.01294.pdf)] A Closer Look at Local Aggregation Operators in Point Cloud Analysis [[github](https://github.com/zeliu98/CloserLook3D)]
- [[NeurIPS2020](https://arxiv.org/pdf/2006.09661.pdf)] Implicit Neural Representations with Periodic Activation Functions [[pytorch](https://github.com/vsitzmann/siren)] :fire:
- [[Arxiv](https://arxiv.org/pdf/2006.07029.pdf)] Rethinking Sampling in 3D Point Cloud Generative Adversarial Networks
- [[Arxiv](https://arxiv.org/pdf/2006.07226.pdf)] Local-Area-Learning Network: Meaningful Local Areas for Efficient Point Cloud Analysis
- [[Arxiv](https://arxiv.org/pdf/2006.10187.pdf)] TearingNet: Point Cloud Autoencoder to Learn Topology-Friendly Representations
- [[Arxiv](https://arxiv.org/pdf/2006.04325.pdf)] Fully Convolutional Mesh Autoencoder using Efficient Spatially Varying Kernels
- [[Arxiv](https://arxiv.org/pdf/2006.07029.pdf)] Rethinking Sampling in 3D Point Cloud Generative Adversarial Networks
- [[Arxiv](https://arxiv.org/pdf/2006.05353.pdf)] MeshWalker: Deep Mesh Understanding by Random Walks
- [[Arxiv](https://arxiv.org/pdf/2005.00383.pdf)] MOPS-Net: A Matrix Optimization-driven Network for Task-Oriented 3D Point Cloud Downsampling
- [[Arxiv](https://arxiv.org/pdf/2004.11784.pdf)] DPDist : Comparing Point Clouds Using Deep Point Cloud Distance
- [[CVPR2020](https://arxiv.org/pdf/2003.00492.pdf)] PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling
- [[AAAI2020](https://arxiv.org/pdf/2004.09411.pdf)] Shape-Oriented Convolution Neural Network for Point Cloud Analysis
- [[Arxiv](https://arxiv.org/pdf/2004.07392.pdf)] Joint Supervised and Self-Supervised Learning for 3D Real-World Challenges
- [[Arxiv](https://arxiv.org/pdf/2004.04462.pdf)] LIGHTCONVPOINT: CONVOLUTION FOR POINTS [[pytorch](https://github.com/A2Zadeh/Variational-Autodecoder)]
- [[Arxiv](https://arxiv.org/pdf/1903.00840.pdf)] Variational Auto-Decoder [[pytorch](https://github.com/A2Zadeh/Variational-Autodecoder)]
- [[Arxiv](https://arxiv.org/pdf/2004.01301.pdf)] Generative PointNet: Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification
- [[CVPR2020](https://arxiv.org/pdf/2004.01002.pdf)] DualConvMesh-Net: Joint Geodesic and Euclidean Convolutions on 3D Meshes [[pytorch](https://github.com/VisualComputingInstitute/dcm-net)]
- [[CVPR2020](https://arxiv.org/pdf/2003.13479.pdf)] RPM-Net: Robust Point Matching using Learned Features [[github](https://github.com/yewzijian/RPMNet)]
- [[CVPR2020](https://arxiv.org/pdf/2003.12971.pdf)] Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds
- [[CVPR2020](https://arxiv.org/pdf/2003.13326.pdf)] PointGMM: a Neural GMM Network for Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2003.10027.pdf)] Dynamic ReLU
- [[CVPR2020](https://arxiv.org/pdf/1912.03663.pdf)] SampleNet: Differentiable Point Cloud Sampling [[pytorch](https://github.com/itailang/SampleNet)]
- [[Arxiv](https://arxiv.org/pdf/2002.11881.pdf)] Defense-PointNet: Protecting PointNet Against Adversarial Attacks
- [[CVPR2020](https://arxiv.org/pdf/2002.10701.pdf)] FPConv: Learning Local Flattening for Point Convolution [[pytorch](https://github.com/lyqun/FPConv)]
- [[SIGGRAPH2019](https://arxiv.org/pdf/1809.05910.pdf)] MeshCNN: A Network with an Edge [[pytorch](https://github.com/ranahanocka/MeshCNN)] :fire::star:
- [[ICCV2019](https://arxiv.org/abs/1904.07615)] Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning [[tensorflow](https://github.com/phermosilla/TotalDenoising)]
- [[ICCV2019](https://arxiv.org/pdf/1907.10844.pdf)] PU-GAN: a Point Cloud Upsampling Adversarial Network:fire:
- [[CVPR2019](https://arxiv.org/pdf/1904.07601.pdf)] Relation-Shape Convolutional Neural Network for Point Cloud Analysis [[pytorch](https://github.com/Yochengliu/Relation-Shape-CNN)] :fire:
- [[CVPR2019](https://arxiv.org/pdf/1811.11286.pdf)] Patch-based Progressive 3D Point Set Upsampling
 [[tensorflow](https://github.com/yifita/3PU)] [[pytorch](https://github.com/yifita/3PU_pytorch)] :fire:
- [[TOG2019](https://arxiv.org/pdf/1801.07829.pdf)] Dynamic Graph CNN for Learning on Point Clouds [[Project](https://liuziwei7.github.io/projects/DGCNN)] :fire: :star:
- [[ECCV2018](https://arxiv.org/pdf/1807.06010.pdf)] EC-Net: an Edge-aware Point set Consolidation Network [[project page](https://yulequan.github.io/ec-net/)]
- [[CVPR2018](https://arxiv.org/pdf/1801.06761.pdf)] PU-Net: Point Cloud Upsampling Network :star::fire:
- [[Arxiv](https://arxiv.org/pdf/2002.10876.pdf)] PointAugment: an Auto-Augmentation Framework for Point Cloud Classification
- [[ICLR2017](https://arxiv.org/pdf/1611.04500.pdf)] DEEP LEARNING WITH SETS AND POINT CLOUDS
- [[NeurIPS2017](http://papers.nips.cc/paper/6931-deep-sets.pdf)] Deep Sets
- [[Siggraph2006](https://www.merl.com/publications/docs/TR2006-054.pdf)] Designing with Distance Fields





---
## Others (inc. Networks in Classification, Matching, Registration, Alignment, Depth, Normal, Pose, Keypoints, etc.)
- [[Arxiv](https://arxiv.org/pdf/2102.09334.pdf)] StablePose: Learning 6D Object Poses from Geometrically Stable Patches
- [[Arxiv](https://arxiv.org/pdf/2102.06697.pdf)] A Parameterised Quantum Circuit Approach to Point Set Matching
- [[Arxiv](https://arxiv.org/pdf/2102.01161.pdf)] Adjoint Rigid Transform Network: Self-supervised Alignment of 3D Shapes
- [[Arxiv](https://arxiv.org/pdf/2102.00719.pdf)] Video Transformer Network
- [[ICLR2021](https://arxiv.org/pdf/2101.12378.pdf)] NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation [[pytorch](https://github.com/Angtian/NeMo)]
- [[Arxiv](https://arxiv.org/pdf/2004.00221v3.pdf)] NBDT: NEURAL-BACKED DECISION TREE [[pytorch](https://github.com/alvinwan/neural-backed-decision-trees)]
- [[Arxiv](https://arxiv.org/pdf/2011.14141v1.pdf)] AdaBins: Depth Estimation using Adaptive Bins [[pytorch](https://github.com/shariqfarooq123/AdaBins)]
- [[Arxiv](https://arxiv.org/pdf/2012.15680.pdf)] Unsupervised Monocular Depth Reconstruction of Non-Rigid Scenes
- [[Arxiv](https://arxiv.org/pdf/2012.15638.pdf)] CorrNet3D: Unsupervised End-to-end Learning of Dense Correspondence for 3D Point Clouds


#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.05877.pdf)] iNeRF: Inverting Neural Radiance Fields for Pose Estimation [[Project](http://yenchenlin.me/inerf/)]
- [[Arxiv](https://arxiv.org/pdf/2012.10296.pdf)] Boosting Monocular Depth Estimation with Lightweight 3D Point Fusion
- [[Arxiv](https://arxiv.org/pdf/2011.11260.pdf)] 3D Registration for Self-Occluded Objects in Context
- [[Arxiv](https://arxiv.org/pdf/2011.12438.pdf)] Continuous Surface Embeddings
- [[Arxiv](https://arxiv.org/pdf/2011.12149.pdf)] SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration
- [[Arxiv](https://arxiv.org/pdf/2011.13244.pdf)] MVTN: Multi-View Transformation Network for 3D Shape Recognition
- [[Arxiv](https://arxiv.org/pdf/2011.13005.pdf)] PREDATOR: Registration of 3D Point Clouds with Low Overlap
- [[Arxiv](https://arxiv.org/pdf/2011.12745.pdf)] Deep Magnification-Arbitrary Upsampling over 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2011.14880.pdf)] Occlusion Guided Scene Flow Estimation on 3D Point Clouds
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf)] An Analysis of SVD for Deep Rotation Estimation 
- [[EG2020W](https://arxiv.org/pdf/2010.09355.pdf)] SHREC 2020 track: 6D object pose estimation
- [[ACCV2020](https://arxiv.org/abs/2010.01912)] Best Buddies Registration for Point Clouds
- [[3DV](https://arxiv.org/pdf/2010.07091.pdf)] A New Distributional Ranking Loss With Uncertainty: Illustrated in Relative Depth Estimation
- [[BMVC2020](https://arxiv.org/pdf/2009.04065.pdf)] View-consistent 4D Light Field Depth Estimation
- [[BMVC2020](https://arxiv.org/pdf/2008.09965.pdf)] Neighbourhood-Insensitive Point Cloud Normal Estimation Network [[Project](http://ninormal.active.vision/)]
- [[ECCV2020](https://arxiv.org/pdf/2008.09088.pdf)] DeepGMR: Learning Latent Gaussian Mixture Models for Registration [[Project](https://wentaoyuan.github.io/deepgmr/)]
- [[ECCV2020](https://arxiv.org/abs/2008.07931)] Motion Capture from Internet Videos [[Project](https://zju3dv.github.io/iMoCap/)]
- [[ECCV2020](https://arxiv.org/pdf/2008.07861.pdf)] Depth Completion with RGB Prior
- [[ECCV2020](https://arxiv.org/pdf/2004.04807.pdf)] 6D Camera Relocalization in Ambiguous Scenes via Continuous Multimodal Inference
- [[Arxiv](https://arxiv.org/pdf/2008.00305.pdf)] Self-Supervised Learning of Point Clouds via Orientation Estimation
- [[SIGGRAPH2020](https://arxiv.org/abs/2008.00485)] SymmetryNet: Learning to Predict Reflectional and Rotational Symmetries of 3D Shapes from Single-View RGB-D Images [[Project](https://kevinkaixu.net/projects/symmetrynet.html#code)]
- [[ECCV2020](https://arxiv.org/pdf/2008.01484.pdf)] Learning Stereo from Single Images [[github](https://github.com/nianticlabs/stereo-from-mono/)]
- [[Arxiv](https://arxiv.org/pdf/2008.02265.pdf)] Learning Long-term Visual Dynamics with Region Proposal Interaction Networks [[Project](https://haozhiqi.github.io/RPIN/)]
- [[ECCV2020](https://arxiv.org/pdf/2008.02004.pdf)] Beyond Controlled Environments: 3D Camera Re-Localization in Changing Indoor Scenes [[Project](https://waldjohannau.github.io/RIO10/)]
- [[ECCV2020](https://arxiv.org/pdf/2007.11341.pdf)] Unsupervised Shape and Pose Disentanglement for 3D Meshes
- [[Arxiv](https://arxiv.org/pdf/2007.07714.pdf)] PVSNet: Pixelwise Visibility-Aware Multi-View Stereo Network
- [[ECCV2020](https://arxiv.org/pdf/2007.07696.pdf)] P<sup>2</sup>Net: Patch-match and Plane-regularization for Unsupervised Indoor Depth Estimation
- [[CVPR2020](https://arxiv.org/pdf/2001.05119.pdf)] Learning multiview 3D point cloud registration [[pytorch](https://github.com/zgojcic/3D_multiview_reg)]
- [[CVPR2020](https://arxiv.org/pdf/2005.01014.pdf)] Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences
- [[Siggraph2020](https://arxiv.org/pdf/2004.15021.pdf)] Consistent Video Depth Estimation
- [[Arxiv](https://arxiv.org/pdf/2004.11563.pdf)] Deep Feature-preserving Normal Estimation for Point Cloud Filtering
- [[Arxiv](https://arxiv.org/pdf/2004.10681.pdf)] Pseudo RGB-D for Self-Improving Monocular SLAM and Depth Prediction
- [[CVPR2020](https://arxiv.org/pdf/2004.01314.pdf)] Towards Better Generalization: Joint Depth-Pose Learning without PoseNet [[pytorch](https://github.com/B1ueber2y/TrianFlow)]
- [[Arxiv](https://arxiv.org/pdf/2004.00740.pdf)] Monocular Camera Localization in Prior LiDAR Maps with 2D-3D Line Correspondences
- [[Arxiv](https://arxiv.org/pdf/2003.08400.pdf)] Adversarial Texture Optimization from RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/2003.08515.pdf)] SAPIEN: A SimulAted Part-based Interactive ENvironment
- [[CVPR2020](https://arxiv.org/pdf/2003.11089.pdf)] G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features
- [[Arxiv](https://arxiv.org/pdf/2003.10664.pdf)] On Localizing a Camera from a Single Image
- [[Arxiv](https://arxiv.org/pdf/2003.10826.pdf)] DeepFit: 3D Surface Fitting via Neural Network Weighted Least Squares
- [[CVPR2020](https://arxiv.org/pdf/2003.10629.pdf)] KFNet: Learning Temporal Camera Relocalization using Kalman Filtering
- [[Arxiv](https://arxiv.org/pdf/2003.10333.pdf)] Neural Contours: Learning to Draw Lines from 3D Shapes
- [[Arxiv](https://arxiv.org/pdf/2003.09175.pdf)] 3dDepthNet: Point Cloud Guided Depth Completion Network for Sparse Depth and Single Color Image
- [[Arxiv](https://arxiv.org/pdf/2003.07619.pdf)] Unsupervised Learning of Category-Specific Symmetric 3D Keypoints from Point Sets
- [[CVPR2020](https://arxiv.org/pdf/2003.05855.pdf)] End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2003.04626.pdf)] PnP-Net: A hybrid Perspective-n-Point Network
- [[CVPR2020](https://arxiv.org/pdf/2003.03522.pdf)] MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision
- [[CVPR2020](https://arxiv.org/pdf/2003.01060.pdf)] D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry
- [[ICIP2020](https://arxiv.org/pdf/2003.00856.pdf)] TRIANGLE-NET: TOWARDS ROBUSTNESS IN POINT CLOUD CLASSIFICATION
- [[ICRA2020](https://arxiv.org/pdf/2003.00188.pdf)] Robust 6D Object Pose Estimation by Learning RGB-D Features
- [[Arxiv](https://arxiv.org/pdf/2002.12730.pdf)] Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation Using Displacement Fields
- [[Arxiv](https://arxiv.org/pdf/2001.05036.pdf)] Single Image Depth Estimation Trained via Depth from Defocus Cues [[pytorch](https://github.com/shirgur/UnsupervisedDepthFromFocus)]
- [[Arxiv](https://arxiv.org/pdf/2001.00987.pdf)] DepthTransfer: Depth Extraction from Video Using Non-parametric Sampling
- [[Arxiv](https://arxiv.org/pdf/1912.12756.pdf)] Target-less registration of point clouds: A review
- [[Arxiv](https://arxiv.org/pdf/1912.12098.pdf)] Quaternion Equivariant Capsule Networks for 3D point clouds
- [[Arxiv](https://arxiv.org/pdf/1912.11913.pdf)] Category-Level Articulated Object Pose Estimation
- [[Arxiv](https://arxiv.org/pdf/1912.12296.pdf)] A Quantum Computational Approach to Correspondence Problems on Point Sets
- [[Arxiv](https://arxiv.org/pdf/1912.09697.pdf)] DeepSFM: Structure From Motion Via Deep Bundle Adjustment
- [[Arxiv](https://arxiv.org/pdf/1912.09316v2.pdf)] P<sup>2</sup>GNet: Pose-Guided Point Cloud Generating Networks for 6-DoF Object Pose Estimation
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Georgakis_Learning_Local_RGB-to-CAD_Correspondences_for_Object_Pose_Estimation_ICCV_2019_paper.pdf)] Learning Local RGB-to-CAD Correspondences for Object Pose Estimation
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dahnert_Joint_Embedding_of_3D_Scan_and_CAD_Objects_ICCV_2019_paper.pdf)] Joint Embedding of 3D Scan and CAD Objects [[dataset](https://github.com/xheon/JointEmbedding)]
- [[ICLR2019](https://arxiv.org/pdf/1806.04807.pdf)] BA-NET: DENSE BUNDLE ADJUSTMENT NETWORKS [[tensorflow](https://github.com/frobelbest/BANet)]
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Grabner_GP2C_Geometric_Projection_Parameter_Consensus_for_Joint_3D_Pose_and_ICCV_2019_paper.pdf)] GP<sup>2</sup>C: Geometric Projection Parameter Consensus for Joint 3D Pose and Focal Length Estimation in the Wild
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Closed-Form_Optimal_Two-View_Triangulation_Based_on_Angular_Errors_ICCV_2019_paper.pdf)] Closed-Form Optimal Two-View Triangulation Based on Angular Errors
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cui_Polarimetric_Relative_Pose_Estimation_ICCV_2019_paper.pdf)] Polarimetric Relative Pose Estimation
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Avetisyan_End-to-End_CAD_Model_Retrieval_and_9DoF_Alignment_in_3D_Scans_ICCV_2019_paper.pdf)] End-to-End CAD Model Retrieval and 9DoF Alignment in 3D Scans
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kong_Deep_Non-Rigid_Structure_From_Motion_ICCV_2019_paper.pdf)] Deep Non-Rigid Structure from Motion
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf)] On the Continuity of Rotation Representations in Neural Networks [[pytorch](https://github.com/papagina/RotationContinuity)]
- [[Arxiv](https://arxiv.org/pdf/1902.10840.pdf)] Deep Interpretable Non-Rigid Structure from Motion [[tensorflow](https://github.com/kongchen1992/deep-nrsfm)]
- [[Arxiv](https://arxiv.org/pdf/1911.07246.pdf)] IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks [[dataset](https://clvrai.github.io/furniture/)]
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Avetisyan_Scan2CAD_Learning_CAD_Model_Alignment_in_RGB-D_Scans_CVPR_2019_paper.pdf)] Scan2CAD: Learning CAD Model Alignment in RGB-D Scans [[pytorch](https://github.com/skanti/Scan2CAD)] :fire:
- [[3DV2019](https://arxiv.org/pdf/1908.02853.pdf)] Location Field Descriptors: Single Image 3D Model Retrieval in the Wild
- [[CVPR2016](http://www.cs.cmu.edu/~aayushb/marrRevisited/aayushb_2d3d.pdf)] Marr Revisited: 2D-3D Alignment via Surface Normal Prediction [[caffe](https://github.com/aayushbansal/MarrRevisited)]







## Survey, Resources and Tools
- [[Survey](https://arxiv.org/pdf/2102.10788.pdf)] Attention Models for Point Clouds in Deep Learning: A Survey
- [[Benchmark](https://arxiv.org/abs/2102.05346)] H3D: Benchmark on Semantic Segmentation of High-Resolution 3D Point Clouds and textured Meshes from UAV LiDAR and Multi-View-Stereo [[Project](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx)]
- [[Survey](https://arxiv.org/pdf/2102.04906.pdf)] Dynamic Neural Networks: A Survey
- [[Survey](https://arxiv.org/pdf/2101.10423v1.pdf)] Online Continual Learning in Image Classification: An Empirical Survey
- [[Survey](https://arxiv.org/pdf/1912.00535v2.pdf)] Deep Learning for Visual Tracking: A Comprehensive Survey
- [[Survey](https://arxiv.org/ftp/arxiv/papers/2101/2101.08845.pdf)] Occlusion Handling in Generic Object Detection: A Review
- [[Survey](https://arxiv.org/abs/2101.10382)] Curriculum Learning: A Survey
- [[Github](https://github.com/yenchenlin/awesome-NeRF)] Awesome Neural Radiance Fields
- [[Survey](https://arxiv.org/pdf/2101.05204.pdf)] Neural Volume Rendering: NeRF And Beyond
- [[Survey](https://arxiv.org/abs/2101.01169)] Transformers in Vision: A Survey
- [[Survey](https://arxiv.org/abs/2009.06732)] Efficient Transformers: A Survey
- [[Survey](https://arxiv.org/abs/1906.01529)] Semantics for Robotic Mapping, Perception and Interaction: A Survey
- [[Survey](https://arxiv.org/abs/1906.01529)] Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy

#### Before 2021
- [[Dataset](https://arxiv.org/pdf/2012.09988.pdf)] Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations [[Github](https://github.com/google-research-datasets/Objectron)]
- [[Survey](https://arxiv.org/pdf/2012.12447.pdf)] Skeleton-based Approaches based on Machine Vision: A Survey
- [[Survey](https://arxiv.org/pdf/2012.13392.pdf)] Deep Learning-Based Human Pose Estimation: A Survey [[Github](https://github.com/zczcwh/DL-HPE)]
- [[Dataset](https://arxiv.org/pdf/2011.02523.pdf)] Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding [[Github](https://github.com/apple/ml-hypersim)]
- [[Survey](https://arxiv.org/pdf/2011.10671.pdf)] A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving [[Github](https://github.com/asharakeh/pod_compare)]
- [[Dataset](https://arxiv.org/pdf/2011.12954.pdf)] RELLIS-3D Dataset: Data, Benchmarks and Analysis [[Github](https://github.com/unmannedlab/RELLIS-3D)]
- [[Arxiv](https://arxiv.org/pdf/2011.03635.pdf)] Motion Prediction on Self-driving Cars: A Review
- [[Github](https://github.com/MIT-TESSE)] TESSE: Unity-based simulator to enable research in perception, mapping, learning, and robotics
- [[Survey](https://arxiv.org/abs/2012.12556)] A Survey on Visual Transformer
- [[Survey](https://arxiv.org/abs/2011.00362)] A Survey on Contrastive Self-supervised Learning
- [[Survey](https://hal.inria.fr/hal-01348404v2/document)] A Survey of Surface Reconstruction from Point Clouds
- [[Dataset](https://arxiv.org/pdf/2010.04642.pdf)] Torch-Points3D: A Modular Multi-Task Framework for Reproducible Deep Learning on 3D Point Clouds [[Project](https://github.com/nicolas-chaulet/torch-points3d)]
- [[Thesis](https://arxiv.org/pdf/2010.09582.pdf)] Learning to Reconstruct and Segment 3D Objects
- [[Survey](https://arxiv.org/abs/2010.15614)] An Overview Of 3D Object Detection
- [[Survey](https://arxiv.org/abs/2010.03978)] A Brief Review of Domain Adaptation
- [[Dataset](https://ai.googleblog.com/2020/11/announcing-objectron-dataset.html)] Announcing the Objectron Dataset
- [[Tutorial](https://arxiv.org/abs/2010.06647)] Video Action Understanding: A Tutorial
- [[Arxiv](https://arxiv.org/abs/2010.02392)] Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Reconstruction [[Page](https://github.com/AutodeskAILab/Fusion360GalleryDataset)]
- [[Survey](https://arxiv.org/abs/2009.09796)] Multi-Task Learning with Deep Neural Networks: A Survey
- [[Survey](https://arxiv.org/pdf/2009.08920.pdf)] Deep Learning for 3D Point Cloud Understanding: A Survey
- [[Thesis](https://arxiv.org/pdf/2009.01786.pdf)] COMPUTATIONAL ANALYSIS OF DEFORMABLE MANIFOLDS: FROM GEOMETRIC MODELING TO DEEP LEARNING
- [[Arxiv](https://arxiv.org/abs/2008.00103)] F*: An Interpretable Transformation of the F-measure
- [[Dataset](http://gibsonenv.stanford.edu/database/)] Gibson Database of 3D Spaces
- [[BMVC2020](https://arxiv.org/abs/2008.05981)] Black Magic in Deep Learning: How Human Skill Impacts Network Training
- [[Arxiv](https://arxiv.org/abs/2008.09164)] PyTorch Metric Learning
- [[Arxiv](https://arxiv.org/abs/2008.00230)] RGB-D Salient Object Detection: A Survey [[Project](https://github.com/taozh2017/RGBD-SODsurvey)]
- [[Arxiv](https://arxiv.org/pdf/2008.01133.pdf)] AiRound and CV-BrCT: Novel Multi-View Datasets for Scene Classification [[Project](http://www.patreo.dcc.ufmg.br/2020/07/22/multi-view-datasets/)]
- [[CVPR2020](https://arxiv.org/pdf/2007.13215.pdf)] OASIS: A Large-Scale Dataset for Single Image 3D in the Wild [[Project](https://oasis.cs.princeton.edu/)]
- [[Arxiv](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)] 3D-FUTURE: 3D FUrniture shape with TextURE
- [[Arxiv](https://arxiv.org/pdf/2011.09127.pdf)] 3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics [[Project](https://pages.tmall.com/wow/cab/tianchi/promotion/alibaba-3d-scene-dataset)][[Link](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)]
- [[Arxiv](https://arxiv.org/pdf/2006.12057.pdf)] Differentiable Rendering: A Survey
- [[Arxiv](https://arxiv.org/pdf/2005.08045.pdf)] Visual Relationship Detection using Scene Graphs: A Survey
- [[Arxiv](https://arxiv.org/pdf/2004.14899.pdf)] Polarization Human Shape and Pose Dataset
- [[Arxiv](https://arxiv.org/pdf/2004.08298.pdf)] IDDA: a large-scale multi-domain dataset for autonomous driving [[Project page](https://idda-dataset.github.io/home/)]
- [[CVPR2020](https://arxiv.org/pdf/2004.06799.pdf)] RoboTHOR: An Open Simulation-to-Real Embodied AI Platform [[Project page](https://ai2thor.allenai.org/robothor/)]
- [[EG2020](https://arxiv.org/pdf/2004.03805.pdf)] State of the Art on Neural Rendering
- [[IJCAI-PRICAI2020](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future?spm=5176.14208320.0.0.1b4f3cf7XdO4pX)] 3D-FUTURE: 3D FUrniture shape with TextURE
- [[Arxiv](https://arxiv.org/pdf/2003.08284.pdf)] Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways
- [[Arxiv](https://arxiv.org/pdf/2002.12687.pdf)] KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations
- [[Arxiv](https://arxiv.org/pdf/2001.06937.pdf)] A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications
- [[Arxiv](https://arxiv.org/pdf/2002.11310.pdf)] From Seeing to Moving: A Survey on Learning for Visual Indoor Navigation (VIN)
- [[Arxiv](https://arxiv.org/abs/1908.00463)] DIODE: A Dense Indoor and Outdoor DEpth Dataset [[dataset](https://github.com/diode-dataset/diode-devkit)]
- [[Github](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch)] Various GANs with Pytorch.
- [[Arxiv](https://arxiv.org/pdf/2002.09147.pdf)] SemanticPOSS: A Point Cloud Dataset with Large Quantity of Dynamic Instances [[dataset](http://www.poss.pku.edu.cn/)]
- [[CVM](https://arxiv.org/pdf/2002.07995.pdf)] A Survey on Deep Geometry Learning: From a Representation Perspective
- [[Arxiv](https://arxiv.org/pdf/2002.08721.pdf)] A survey on Semi-, Self- and Unsupervised Techniques in Image Classification
- [[Arxiv](https://arxiv.org/pdf/2002.04688.pdf)] fastai: A Layered API for Deep Learning
- [[Arxiv](https://arxiv.org/pdf/2001.11737.pdf)] AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance [[dataset](https://bozcani.github.io/auairdataset)]
- [[Arxiv](https://arxiv.org/pdf/2001.10773.pdf)] VIRTUAL KITTI 2 [[dataset](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)]
- [[Arxiv](https://arxiv.org/pdf/1606.05908.pdf)] Tutorial on Variational Autoencoders
- [[Arxiv](https://arxiv.org/pdf/2001.06280.pdf)] Review: deep learning on 3D point clouds
- [[Arxiv](https://arxiv.org/pdf/2001.05566.pdf)] Image Segmentation Using Deep Learning: A Survey
- [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/html/Shin_Pixels_Voxels_and_CVPR_2018_paper.html)] Pixels, Voxels, and Views: A Study of Shape Representations for Single View 3D Object Shape Prediction
- [[Arxiv](https://arxiv.org/pdf/2001.04074.pdf)] Evolution of Image Segmentation using Deep Convolutional Neural Network: A Survey
- [[Arxiv](https://arxiv.org/pdf/2001.01788.pdf)] MCMLSD: A Probabilistic Algorithm and Evaluation Framework for Line Segment Detection
- [[Arxiv](https://arxiv.org/pdf/1912.12033.pdf)] Deep Learning for 3D Point Clouds: A Survey
- [[Arxiv](https://arxiv.org/pdf/1912.10230.pdf)] A Survey on Deep Learning-based Architectures for Semantic Segmentation on 2D images
- [[Arxiv](https://arxiv.org/pdf/1906.06113.pdf)] A Survey on Deep Learning Architectures for Image-based Depth Reconstruction
- [[Arxiv](https://arxiv.org/pdf/1912.10013.pdf)] secml: A Python Library for Secure and Explainable Machine Learning
- [[Arxiv](https://arxiv.org/pdf/1912.03858.pdf)] Bundle Adjustment Revisited
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bi_Deep_CG2Real_Synthetic-to-Real_Translation_via_Image_Disentanglement_ICCV_2019_paper.pdf)] Deep CG2Real: Synthetic-to-Real Translation via Image Disentanglement
- [[Arxiv](https://arxiv.org/pdf/1608.01807.pdf)] SIFT Meets CNN:
A Decade Survey of Instance Retrieval
- [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Uy_Revisiting_Point_Cloud_Classification_A_New_Benchmark_Dataset_and_Classification_ICCV_2019_paper.pdf)] Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data [[tensorflow](https://github.com/hkust-vgd/scanobjectnn)]
- [[Arxiv](https://arxiv.org/pdf/1911.10127.pdf)] BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks [[dataset](https://github.com/YoYo000/BlendedMVS)]
- [[Arxiv](https://arxiv.org/pdf/1909.00169.pdf)] Imbalance Problems in Object Detection: A Review [[repository](https://github.com/kemaloksuz/ObjectDetectionImbalance)]
- [[IJCV](https://link.springer.com/content/pdf/10.1007/s11263-019-01247-4.pdf)] Deep Learning for Generic Object Detection: A Survey
- [[Arxiv](https://arxiv.org/pdf/1904.12228.pdf)] Differentiable Visual Computing (Ph.D thesis)
- [[BMVC2018](https://interiornet.org/items/interiornet_paper.pdf)] InteriorNet: Mega-scale Multi-sensor Photo-realistic Indoor Scenes Dataset [[dataset](https://interiornet.org/)]
- [[ICCV2017](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237796&tag=1)] The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes [[dataset](https://www.mapillary.com/dataset/vistas?pKey=1GyeWFxH_NPIQwgl0onILw)] [[script](https://github.com/mapillary/mapillary_vistas)] :star:
- [[Arxiv](https://arxiv.org/pdf/1907.04758.pdf)] SynthCity: A large scale synthetic point cloud [[dataset](http://www.synthcity.xyz/)]
- [[Github](https://github.com/davidstutz/mesh-voxelization)] Mesh Voxelization (SDFs or Occupancy grids)
- [[Github](https://github.com/christopherbatty/SDFGen)] SDFGen (to generate grid-based signed distance field (level set))
- [[Github](https://github.com/davidstutz/bpy-visualization-utils)] Blender renderer for python
- [[Github](https://github.com/weiaicunzai/blender_shapenet_render)] Blender renderer for python
- [[Github](https://github.com/andyzeng/tsdf-fusion-python)] Volumetric TSDF Fusion of RGB-D Images in Python
- [[Github](https://github.com/andyzeng/tsdf-fusion)] Volumetric TSDF Fusion of Multiple Depth Maps
- [[Github](https://github.com/griegler/pyfusion)] PyFusion
- [[Github](https://github.com/griegler/pyrender)] PyRender
- [[Github](https://github.com/pmneila/PyMCubes)] PyMCubes
- [[Github](https://github.com/autonomousvision/occupancy_networks/tree/b72f6fcd5f3f4761d261805edab0262604f967ae/external/mesh-fusion)] Watertight and Simplified Meshes through TSDF Fusion (Python tool for obtaining watertight meshes using TSDF fusion.)
- [[Github](https://github.com/davidstutz/aml-improved-shape-completion)] Several tools about SDF functions.
- [[Github](https://github.com/andyzeng/3dmatch-toolbox#multi-frame-depth-tsdf-fusion)] 3DMatch Toolbox
- [[stackoverflow](https://stackoverflow.com/questions/44066118/computing-truncated-signed-distance-functiontsdf-from-a-point-cloud)] Computing truncated signed distance function(TSDF) from a point cloud
- [[Github](https://github.com/ethz-asl/voxblox)] voxblox: A library for flexible voxel-based mapping, mainly focusing on truncated and Euclidean signed distance fields.
- [[Github](https://github.com/InteractiveComputerGraphics/Discregrid)] Discregrid: A static C++ library for the generation of discrete functions on a box-shaped domain. This is especially suited for the generation of signed distance fields.
- [[Github](https://github.com/meshula/awesome-voxel#sparse-volumes)] awesome-voxel: Voxel resources for coders
- [[Github](https://github.com/NVIDIA/gvdb-voxels)] gvdb-voxels: Sparse volume compute and rendering on NVIDIA GPUs
- [[Github](https://github.com/daavoo/pyntcloud)] pyntcloud is a Python library for working with 3D point clouds.
- [[Github](http://www.open3d.org/docs/release/index.html)] Open3D: A Modern Library for 3D Data Processing
- [[Github](https://github.com/marian42/mesh_to_sdf)] mesh_to_sdf: Calculate signed distance fields for arbitrary meshes

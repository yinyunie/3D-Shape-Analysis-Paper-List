# 3D-Shape-Analysis-Paper-List
A list of papers, libraries and datasets I recently read is collected for anyone who shows interest at 

---
- [3D Detection & Segmentation](#3d-detection--segmentation)
- [Shape Representation](#shape-representation)
- [Shape & Scene Completion](#shape--scene-completion)
- [Shape Reconstruction](#shape-reconstruction--generation)
- [3D Scene Understanding](#3d-scene-understanding)
- [3D Scene Reconstruction](#3d-scene-reconstruction)
- [NeRF](#nerf)
- [About Human Body](#about-human-body)
- [General Methods](#general-methods)
- [Others (inc. Networks in Classification, Matching, Registration, Alignment, Depth, Normal, Pose, Keypoints, etc.)](#others-inc-networks-in-classification-matching-registration-alignment-depth-normal-pose-keypoints-etc)
- [Survey, Resources and Tools](#survey-resources-and-tools)
---


Statistics: :fire: code is available & stars >= 100 &emsp;|&emsp; :star: citation >= 50




## 3D Detection & Segmentation
- [[ECCV2022](https://arxiv.org/abs/2207.06985)] ObjectBox: From Centers to Boxes for Anchor-Free Object Detection [[github](https://github.com/MohsenZand/ObjectBox)]
- [[Arxiv](https://arxiv.org/abs/2207.00531)] Masked Autoencoders for Self-Supervised Learning on Automotive Point Clouds
- [[CVPR2022](https://arxiv.org/abs/2204.05599)] HyperDet3D: Learning a Scene-conditioned 3D Object Detector
- [[Arxiv](https://arxiv.org/abs/2201.06493v1)] AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection
#### Before 2022
- [[AAAI2022](https://arxiv.org/abs/2112.09205v1)] AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds
- [[AAAI2022](https://arxiv.org/abs/2112.07241v1)] Static-Dynamic Co-Teaching for Class-Incremental 3D Object Detection
- [[NeurIPS2021](https://arxiv.org/abs/2112.07787v1)] Revisiting 3D Object Detection From an Egocentric Perspective
- [[Arxiv](https://arxiv.org/abs/2112.06375v1)] Embracing Single Stride 3D Object Detector with Sparse Transformer [[github](https://github.com/TuSimple/SST)]
- [[AAAI2022](https://arxiv.org/abs/2112.04628v1)] Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2112.04764v1)] 3D-VField: Learning to Adversarially Deform Point Clouds for Robust 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2112.04702v1)] Fast Point Transformer
- [[3DV2021](https://arxiv.org/abs/2112.01135v1)] Open-set 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2112.00322v1)] FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection [[Project](https://github.com/samsunglabs/fcaf3d)]
- [[TPAMI2021](https://arxiv.org/abs/2111.15210v1)] Point Cloud Instance Segmentation with Semi-supervised Bounding-Box Mining
- [[Arxiv](https://arxiv.org/abs/2111.12728v1)] Online Adaptation for Implicit Object Tracking and Shape Reconstruction in the Wild
- [[Arxiv](https://arxiv.org/abs/2111.09515v1)] RAANet: Range-Aware Attention Network for LiDAR-based 3D Object Detection with Auxiliary Density Level Estimation [[github](https://github.com/anonymous0522/RAAN)]
- [[Arxiv](https://arxiv.org/abs/2111.09621v1)] SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking [[github](https://github.com/TuSimple/SimpleTrack)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.06881v1)] Multimodal Virtual Point 3D Detection [[Project](https://tianweiy.github.io/mvp/)]
- [[BMVC2021](https://arxiv.org/abs/2110.14921v1)] 3D Object Tracking with Transformer [[github](https://github.com/3bobo/lttr)]
- [[3DV2021](https://arxiv.org/abs/2110.11325v1)] Learning 3D Semantic Segmentation with only 2D Image Supervision
- [[3DV2021](https://arxiv.org/abs/2110.09936v1)] NeuralDiff: Segmenting 3D objects that move in egocentric videos [[Project](https://www.robots.ox.ac.uk/~vadim/neuraldiff/)]
- [[BMVC2021](https://arxiv.org/abs/2110.09355v1)] FAST3D: Flow-Aware Self-Training for 3D Object Detectors
- [[ICCV2021](https://arxiv.org/abs/2110.08188v1)] Guided Point Contrastive Learning for Semi-supervised Point Cloud Semantic Segmentation
- [[CORL2021](https://arxiv.org/abs/2110.06922v1)] DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries [[github](https://github.com/WangYueFt/detr3d)]
- [[NeurIPS2021](https://arxiv.org/abs/2110.06923v1)] Object DGCNN: 3D Object Detection using Dynamic Graphs [[github](https://github.com/WangYueFt/detr3d)]
- [[Arxiv](https://arxiv.org/abs/2110.06049v1)] Improved Pillar with Fine-grained Feature for 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2110.02531v1)] 3D-FCT: Simultaneous 3D Object Detection and Tracking Using Feature Correlation
- [[ICCVW2021](https://arxiv.org/abs/2110.00464v1)] MonoCInIS: Camera Independent Monocular 3D Object Detection using Instance Segmentation
- [[Arxiv](https://arxiv.org/abs/2109.11835v1)] GSIP: Green Semantic Segmentation of Large-Scale Indoor Point Clouds
- [[Arxiv](https://arxiv.org/abs/2109.10852v1)] Pix2seq: A Language Modeling Framework for Object Detection
- [[Arxiv](https://arxiv.org/abs/2109.10473v1)] MVM3Det: A Novel Method for Multi-view Monocular 3D Detection
- [[ICCV2021](https://arxiv.org/abs/2109.04456v1)] NEAT: Neural Attention Fields for End-to-End Autonomous Driving [[github](https://github.com/autonomousvision/neat)]
- [[ICCV2021](https://arxiv.org/abs/2109.02499v1)] Pyramid R-CNN: Towards Better Performance and Adaptability for 3D Object Detection
- [[ICCV2021](https://arxiv.org/abs/2109.01066v1)] 4D-Net for Learned Multi-Modal Alignment
- [[ICCV2021](https://arxiv.org/abs/2103.16130)] Active Learning for Deep Object Detection via Probabilistic Modeling [[github](https://github.com/nvlabs/al-mdn)]
- [[ICCV2021](https://arxiv.org/abs/2109.08141)] An End-to-End Transformer Model for 3D Object Detection [[Project](https://facebookresearch.github.io/3detr/)]
- [[ICCV2021](https://arxiv.org/abs/2108.10723)] Improving 3D Object Detection with Channel-wise Transformer
- [[ICCV2021](https://arxiv.org/abs/2109.02497)] Voxel Transformer for 3D Object Detection
- [[CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chai_To_the_Point_Efficient_3D_Object_Detection_in_the_Range_CVPR_2021_paper.pdf)] To the Point: Efficient 3D Object Detection in the Range Image With Graph Convolution Kernels
- [[Arxiv](https://arxiv.org/abs/2104.11896)] M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers
- [[ICCV2021](https://arxiv.org/abs/2108.10312v1)] Exploring Simple 3D Multi-Object Tracking for Autonomous Driving
- [[ICCV2021](https://arxiv.org/abs/2108.08258v1)] LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-based 3D Detector
- [[ICCV2021](https://arxiv.org/abs/2108.07478v1)] Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks [[github](https://github.com/Gorilla-Lab-SCUT/SSTNet)]
- [[ICCV2021](https://arxiv.org/abs/2108.07794v1)] RandomRooms: Unsupervised Pre-training from Synthetic Shapes and Randomized Layouts for 3D Object Detection
- [[ICCV2021](https://arxiv.org/abs/2108.06417v1)] Is Pseudo-Lidar needed for Monocular 3D Object detection?
- [[IROS2021](https://arxiv.org/abs/2108.06455v1)] PTT: Point-Track-Transformer Module for 3D Single Object Tracking in Point Clouds [[github](https://github.com/shanjiayao/PTT)]
- [[ICCV2021](https://arxiv.org/abs/2108.05699v1)] Oriented R-CNN for Object Detection [[github](https://github.com/jbwang1997/OBBDetection)]
- [[ICCV2021](https://arxiv.org/abs/2108.04728v1)] Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds [[github](https://github.com/Ghostish/BAT)]
- [[IROS2021](https://arxiv.org/abs/2108.04602v1)] Joint Multi-Object Detection and Tracking with Camera-LiDAR Fusion for Autonomous Driving
- [[ACMMM2021](https://arxiv.org/abs/2108.03648v1)] From Voxel to Point: IoU-guided 3D Object Detection for Point Cloud with Voxel-to-Point Decoder [[github](https://github.com/jialeli1/From-Voxel-to-Point)]
- [[ICCV2021](https://arxiv.org/abs/2108.04023v1)] DRINet: A Dual-Representation Iterative Learning Network for Point Cloud Segmentation
- [[ICCV2021](https://arxiv.org/abs/2108.02350v1)] Hierarchical Aggregation for 3D Instance Segmentation [[github](https://github.com/hustvl/HAIS)]
- [[Arxiv](https://arxiv.org/abs/2108.00620v1)] Investigating Attention Mechanism in 3D Point Cloud Object Detection [[pytorch](https://github.com/ShiQiu0419/attentions_in_3D_detection)]
- [[ICCV2021](https://arxiv.org/abs/2107.13824v1)] VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation [[pytorch](https://github.com/hzykent/VMNet)]
- [[ICCV2021](https://arxiv.org/abs/2107.13774v1)] Geometry Uncertainty Projection Network for Monocular 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2107.13269v1)] Aug3D-RPN: Improving Monocular 3D Object Detection by Synthetic Images with Virtual Depth
- [[Arxiv](https://arxiv.org/abs/2107.12707v1)] DV-Det: Efficient 3D Point Cloud Object Detection with Dynamic Voxelization
- [[ICCV2021](https://arxiv.org/abs/2107.11769v1)] ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation
- [[ICCV2021](https://arxiv.org/abs/2107.11669)] Rank &amp; Sort Loss for Object Detection and Instance Segmentation [[pytorch](https://github.com/kemaloksuz/RankSortLoss)]
- [[Arxiv](https://arxiv.org/abs/2107.04013v1)] Multi-Modality Task Cascade for 3D Object Detection [[github](https://github.com/Divadi/MTC_RCNN)]
- [[ACMMM2021](https://arxiv.org/abs/2107.02493v1)] Neighbor-Vote: Improving Monocular 3D Object Detection through Neighbor Distance Voting
- [[Arxiv](https://arxiv.org/abs/2106.15796v1)] Monocular 3D Object Detection: An Extrinsic Parameter Free Approach
- [[Arxiv](https://arxiv.org/pdf/2106.14101v1.pdf)] Real-time 3D Object Detection using Feature Map Flow [[pytorch](https://github.com/YoushaaMurhij/FMFNet)]
- [[Arxiv](https://arxiv.org/pdf/2106.13381v1.pdf)] To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels
- [[CVPR2021](https://arxiv.org/abs/2106.13365v1)] RSN: Range Sparse Net for Efficient, Accurate LiDAR 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2106.06882v1)] Sparse PointPillars: Exploiting Sparsity in Birds-Eye-View Object Detection
- [[Arxiv](https://arxiv.org/abs/2106.01178)] ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection [[Project](https://github.com/saic-vul/imvoxelnet)]
- [[CVPR2021](https://arxiv.org/pdf/2105.06461.pdf)] 3D Spatial Recognition without Spatially Labeled 3D [[Project](https://facebookresearch.github.io/WyPR/)]
- [[Arxiv](https://arxiv.org/abs/2105.00268)] Lite-FPN for Keypoint-based Monocular 3D Object Detection
- [[TPAMI](https://arxiv.org/abs/2104.08797)] MonoGRNet: A General Framework for Monocular 3D Object Detection
- [[Arxiv](https://arxiv.org/abs/2104.09035)] Lidar Point Cloud Guided Monocular 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2104.05858.pdf)] Geometry-aware data augmentation for monocular 3D object detection
- [[Arxiv](https://arxiv.org/abs/2104.06041)] OCM3D: Object-Centric Monocular 3D Object Detection
- [[CVPR2021](https://arxiv.org/abs/2104.02323v1)] Objects are Different: Flexible Monocular 3D Object Detection [[github](https://github.com/zhangyp15/MonoFlex)]
- [[CVPR2021](https://arxiv.org/abs/2104.00902v1)] HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection
- [[Arxiv](https://arxiv.org/pdf/2104.00678v1.pdf)] Group-Free 3D Object Detection via Transformers
 [[pytorch](https://github.com/zeliu98/Group-Free-3D)]
- [[CVPR2021](https://arxiv.org/pdf/2103.17202v1.pdf)] GrooMeD-NMS: Grouped Mathematically Differentiable NMS for Monocular 3D Object Detection [[pytorch](https://github.com/abhi1kumar/groomed_nms)]
- [[CVPR2021](https://arxiv.org/pdf/2104.06114.pdf)] Back-tracing Representative Points for Voting-based 3D Object Detection in Point Clouds [[pytorch](https://github.com/cheng052/BRNet)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16470v1.pdf)] Depth-conditioned Dynamic Message Propagation for Monocular 3D Object Detection [[github](https://github.com/fudan-zvg/DDMP)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16237v1.pdf)] Delving into Localization Errors for Monocular 3D Object Detection [[github](https://github.com/xinzhuma/monodle)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16054v1.pdf)] 3D-MAN: 3D Multi-frame Attention Network for Object Detection
- [[CVPR2021](https://arxiv.org/pdf/2103.15297v1.pdf)] LiDAR R-CNN: An Efficient and Universal 3D Object Detector [[github](https://github.com/tusimple/LiDAR_RCNN)]
- [[CVPR2021](https://arxiv.org/pdf/2012.04355v2.pdf)] 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection [[pytorch](https://github.com/THU17cyz/3DIoUMatch)]
- [[CVPR2021](https://arxiv.org/pdf/2103.13164v1.pdf)] M3DSSD: Monocular 3D Single Stage Object Detector
- [[CVPR2021](https://arxiv.org/abs/2103.12605v2)] MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation
- [[Arxiv](https://arxiv.org/pdf/2103.10042.pdf)] SparsePoint: Fully End-to-End Sparse 3D Object Detector
- [[Arxiv](https://arxiv.org/abs/2103.10039)] RangeDet:In Defense of Range View for LiDAR-based 3D Object Detection
- [[ICRA2021](https://arxiv.org/abs/2103.09422)] YOLOStereo3D: A Step Back to 2D for Efficient Stereo 3D Detection [[github](https://github.com/Owen-Liuyuxuan/visualDet3D)]
- [[CVPR2021](https://arxiv.org/pdf/2103.05346v1.pdf)] ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection [[github](https://github.com/CVMI-Lab/ST3D)]
- [[Arxiv](https://arxiv.org/pdf/2103.05073v1.pdf)] Offboard 3D Object Detection from Point Cloud Sequences
- [[CVPR2021](https://arxiv.org/abs/2011.13328v2)] DyCo3D: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution [[github](https://github.com/aim-uofa/DyCo3D)]
- [[Arxiv](https://arxiv.org/pdf/2103.02093.pdf)] Pseudo-labeling for Scalable 3D Object Detection
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
- [[Arxiv](https://arxiv.org/pdf/2006.05682.pdf)] H3DNet: 3D Object Detection Using Hybrid Geometric Primitives [[pytorch](https://github.com/zaiweizhang/H3DNet)]
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
- [[ECCV2022](https://arxiv.org/abs/2207.11911)] NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing [[Project](https://zju3dv.github.io/neumesh/)]
- [[Arxiv](https://arxiv.org/abs/2207.01545)] Masked Autoencoders in 3D Point Cloud Representation Learning
- [[Arxiv](https://arxiv.org/abs/2206.05837v1)] NeuralODF: Learning Omnidirectional Distance Fields for 3D Shape Representation
- [[Siggraph2022](https://arxiv.org/abs/2202.08345)] Learning Smooth Neural Functions via Lipschitz Regularization [[Project](https://nv-tlabs.github.io/lip-mlp/)]
- [[Siggraph2022](https://arxiv.org/abs/2205.02825)] Dual Octree Graph Networks for Learning Adaptive Volumetric Shape Representations [[Project](https://wang-ps.github.io/dualocnn)]
- [[Arxiv](https://arxiv.org/abs/2204.07159)] A Level Set Theory for Neural Implicit Evolution under Explicit Flows
- [[CVPR2022](https://arxiv.org/abs/2204.07126)] GIFS: Neural Implicit Function for General Shape Representation [[Project](https://jianglongye.com/gifs/)]
- [[Arxiv](https://arxiv.org/abs/2202.04713v1)] PINs: Progressive Implicit Networks for Multi-Scale Neural Representations
- [[Arxiv](https://arxiv.org/abs/2202.04241v1)] Distillation with Contrast is All You Need for Self-Supervised Point Cloud Representation Learning
- [[Arxiv](https://arxiv.org/abs/2202.02444v1)] Spelunking the Deep: Guaranteed Queries for General Neural Implicit Surfaces
- [[Arxiv](https://arxiv.org/abs/2202.03532)] MINER: Multiscale Implicit Neural Representations
- [[Arxiv](https://arxiv.org/abs/2201.02279v1)] De-rendering 3D Objects in the Wild
- [[Arxiv](https://arxiv.org/abs/2201.00785v1)] Implicit Autoencoder for Point Cloud Self-supervised Representation Learning
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.10196v1)] End-to-End Learning of Multi-category 3D Pose and Shape Estimation
- [[Arxiv](https://arxiv.org/abs/2112.09329v1)] Point2Cyl: Reverse Engineering 3D Objects from Point Clouds to Extrusion Cylinders
- [[Arxiv](https://arxiv.org/abs/2112.05300v1)] Representing 3D Shapes with Probabilistic Directed Distance Fields
- [[Arxiv](https://arxiv.org/abs/2112.03221v1)] Text2Mesh: Text-Driven Neural Stylization for Meshes [[Project](https://threedle.github.io/text2mesh/)]
- [[Arxiv](https://arxiv.org/abs/2112.02413v1)] PointCLIP: Point Cloud Understanding by CLIP [[github](https://github.com/ZrrSkywalker/PointCLIP)]
- [[Arxiv](https://arxiv.org/abs/2111.15363v1)] Voint Cloud: Multi-View Point Cloud Representation for 3D Understanding
- [[Arxiv](https://arxiv.org/abs/2111.13652v1)] Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction
- [[Arxiv](https://arxiv.org/abs/2111.12488v1)] Intuitive Shape Editing in Latent Space
- [[NeurIPS2021](https://arxiv.org/abs/2111.07117v1)] Learning Object-Centric Representations of Multi-Object Scenes from Multiple Views [[github](https://github.com/NanboLi/MulMON)]
- [[Arxiv](https://arxiv.org/abs/2111.13674)] Neural Fields as Learnable Kernels for 3D Reconstruction
- [[NeurIPS2021](https://arxiv.org/abs/2111.01067v1)] OctField: Hierarchical Implicit Functions for 3D Modeling [[github](https://github.com/IGLICT/OctField)]
- [[3DV2021](https://arxiv.org/abs/2110.11036v1)] RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation [[github](https://github.com/CVLAB-Unibo/RefRec)]
- [[3DV2021](https://arxiv.org/abs/2110.07882v1)] PolyNet: Polynomial Neural Network for 3D Shape Recognition with PolyShape Representation [[Project](https://arxiv.org/pdf/2110.07882v1.pdf)]
- [[Arxiv](https://arxiv.org/abs/2112.04645)] BACON: Band-limited Coordinate Networks for Multiscale Scene Representation [[Project](https://davidlindell.com/publications/bacon)]
- [[Arxiv](https://arxiv.org/abs/2112.05381)] UNIST: Unpaired Neural Implicit Shape Translation Network [[Project](https://qiminchen.github.io/unist/)]
- [[Arxiv](https://arxiv.org/abs/2109.01605v1)] Representing Shape Collections with Alignment-Aware Linear Models [[Project](https://romainloiseau.github.io/deep-linear-shapes/)]
- [[ICCV2021](https://arxiv.org/abs/2109.00179v1)] Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds
- [[Arxiv](https://arxiv.org/abs/2111.09383?context=cs.GR)] DeepCurrents: Learning Implicit Representations of Shapes with Boundaries
- [[3DV](https://arxiv.org/abs/2110.11860)] AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations [[github](https://github.com/SimonGiebenhain/AIR-Nets)]
- [[Arxiv](https://arxiv.org/abs/2110.05770)] HyperCube: Implicit Field Representations of Voxelized 3D Models
- [[Arxiv](https://arxiv.org/abs/2108.09432v1)] ARAPReg: An As-Rigid-As Possible Regularization Loss for Learning Deformable Shape Generators
- [[ICCV2021](https://arxiv.org/abs/2109.05591)] Multiresolution Deep Implicit Functions for 3D Shape Representation
- [[ICCV2021](https://arxiv.org/abs/2108.04628v1)] Learning Canonical 3D Object Representation for Fine-Grained Recognition
- [[Arxiv](https://arxiv.org/abs/2108.02104v1)] Point Discriminative Learning for Unsupervised Representation Learning on 3D Point Clouds
- [[Arxiv](https://arxiv.org/abs/2107.11024)] A Deep Signed Directional Distance Function for Object Shape Representation
- [[Arxiv](https://arxiv.org/abs/2107.04004v1)] 3D Neural Scene Representations for Visuomotor Control [[Project](https://3d-representation-learning.github.io/nerf-dy/)]
- [[Arxiv](https://arxiv.org/pdf/2104.07645.pdf)] A-SDF: Learning Disentangled Signed Distance Functions
for Articulated Shape Representation [[Project](https://jitengmu.github.io/A-SDF/)]
- [[Arxiv](https://arxiv.org/abs/2104.06392)] ShapeMOD: Macro Operation Discovery for 3D Shape Programs [[Project](https://rkjones4.github.io/shapeMOD.html)]
- [[Arxiv](https://arxiv.org/pdf/2104.03851v1.pdf)] CoCoNets: Continuous Contrastive 3D Scene Representations [[Project](https://mihirp1998.github.io/project_pages/coconets/)]
- [[Arxiv](https://arxiv.org/abs/2102.09105)] DeepMetaHandles: Learning Deformation Meta-Handles of 3D Meshes with Biharmonic Coordinates [[Project](https://github.com/Colin97/DeepMetaHandles)]

#### Before 2021
- [[CVPR2021](http://campar.in.tum.de/pub/paetzold2021cldice/paetzold2021cldice.pdf)] clDice-a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation [[github](https://github.com/jocpae/clDice)]
- [[CVPR2021](https://arxiv.org/pdf/2012.00230.pdf)] Point2Skeleton: Learning Skeletal Representations from Point Clouds [[pytorch](https://github.com/clinplayer/Point2Skeleton)]
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
- [[ECCV2022](https://arxiv.org/abs/2207.11467)] CompNVS: Novel View Synthesis with Scene Completion
- [[ECCV2022](https://arxiv.org/abs/2207.11790)] PatchRD: Detail-Preserving Shape Completion by Learning Patch Retrieval and Deformation [[Project](https://github.com/GitBoSun/PatchRD)]
- [[Arxiv](https://arxiv.org/abs/2202.02669v1)] SRPCN: Structure Retrieval based Point Completion Network
- [[ICRA2022](https://arxiv.org/abs/2202.03084v1)] Temporal Point Cloud Completion with Pose Disturbance
- [[Arxiv](https://arxiv.org/abs/2201.01858v1)] Towards realistic symmetry-based completion of previously unseen point clouds [[github](https://github.com/softserveinc-rnd/symmetry-3d-completion)]
#### Before 2022
- [[AAAI2022](https://arxiv.org/abs/2112.12925v1)] Not All Voxels Are Equal: Semantic Scene Completion from the Point-Voxel Perspective
- [[AAAI2022](https://arxiv.org/abs/2112.05324v1)] Attention-based Transformation from Latent Features to Point Clouds
- [[Arxiv](https://arxiv.org/abs/2112.00726v1)] MonoScene: Monocular 3D Semantic Scene Completion [[Project](https://github.com/cv-rits/MonoScene)]
- [[Arxiv](https://arxiv.org/abs/2111.14798v1)] Semi-supervised Implicit Scene Completion from Sparse LiDAR [[github](https://github.com/OPEN-AIR-SUN/SISC)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.12702v1)] Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion [[github](https://github.com/wutong16/Density_aware_Chamfer_Distance)]
- [[Arxiv](https://arxiv.org/abs/2111.12242v1)] PU-Transformer: Point Cloud Upsampling Transformer
- [[BMVC2021](https://arxiv.org/abs/2111.10701v1)] Self-Supervised Point Cloud Completion via Inpainting
- [[IROS2021](https://arxiv.org/abs/2112.01840v1)] Graph-Guided Deformation for Point Cloud Completion
- [[IROS2021](https://arxiv.org/abs/2109.11453v1)] Semantic Segmentation-assisted Scene Completion for LiDAR Point Clouds [[github](https://github.com/jokester-zzz/SSA-SC)]
- [[Arxiv](https://arxiv.org/abs/2109.10161v1)] 3D Point Cloud Completion with Geometric-Aware Adversarial Augmentation
- [[Arxiv](https://arxiv.org/abs/2109.09337v1)] PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling
- [[ICCV2021](https://arxiv.org/abs/2108.09936v1)] Voxel-based Network for Shape Completion by Leveraging Edge Generation [[github](https://github.com/xiaogangw/VE-PCN)]
- [[ICCV2021](https://arxiv.org/abs/2108.08839v1)] PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers [[github](https://github.com/yuxumin/PoinTr)]
- [[ICCV2021](https://arxiv.org/abs/2108.04444v1)] SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer [[github](https://github.com/AllenXiangX/SnowflakeNet)]
- [[Arxiv](https://arxiv.org/abs/2107.13452v1)] CarveNet: Carving Point-Block for Complex 3D Shape Completion
- [[IJCAI2021](https://arxiv.org/abs/2106.15413v1)] IMENet: Joint 3D Semantic Scene Completion and 2D Semantic Segmentation through Iterative Mutual Enhancement
- [[CVPR2021](https://arxiv.org/abs/2106.04779v1)] Point Cloud Upsampling via Disentangled Refinement [[github](https://github.com/liruihui/Dis-PU)]
- [[TVCG2021](https://arxiv.org/abs/2106.00329)] Consistent Two-Flow Network for Tele-Registration of Point Clouds [[Project](https://vcc.tech/research/2021/CTFNet)]
- [[Arxiv](https://arxiv.org/abs/2105.01905)] 4DComplete: Non-Rigid Motion Estimation Beyond the Observable Surface [[Project](https://github.com/rabbityl/DeformingThings4D)]
- [[CVPR2021](https://arxiv.org/pdf/2104.13366.pdf)] Unsupervised 3D Shape Completion through GAN Inversion [[Project](https://junzhezhang.github.io/projects/ShapeInversion/)]
- [[Arxiv](https://arxiv.org/abs/2104.09587)] ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion
- [[CVPR2021](https://arxiv.org/pdf/2104.10154.pdf)] Variational Relational Point Completion Network [[Project](https://paul007pl.github.io/projects/VRCNet)]
- [[CVPR2021](https://arxiv.org/abs/2104.05666)] View-Guided Point Cloud Completion
- [[CVPR2021](https://arxiv.org/pdf/2104.03640v1.pdf)] Semantic Scene Completion via Integrating Instances and Scene in-the-Loop [[pytorch](https://github.com/yjcaimeow/SISNet)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16671v1.pdf)] Denoise and Contrast for Category Agnostic Shape Completion
- [[CVPR2021](https://arxiv.org/pdf/2103.07838.pdf)] Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding
- [[CVPR2021](https://arxiv.org/abs/2012.03408v2)] PMP-Net: Point Cloud Completion by Learning Multi-step Point Moving Paths
- [[CVPR2021](https://arxiv.org/pdf/2103.02535.pdf)] Style-based Point Generator with Adversarial Rendering for Point Cloud Completion
- [[Arxiv](https://arxiv.org/pdf/2008.03404.pdf)] VPC-Net: Completion of 3D Vehicles from MLS Point Clouds

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
## Shape Reconstruction & Generation
- [[ECCV2022](https://arxiv.org/abs/2207.11795)] Cross-Modal 3D Shape Generation and Manipulation [[Project](https://people.cs.umass.edu/~zezhoucheng/edit3d/)]
- [[ECCV2022](https://arxiv.org/abs/2207.12298)] Deforming Radiance Fields with Cages
- [[NeurIPS2021](https://arxiv.org/abs/2110.07604)] NeRS: Neural Reflectance Surfaces for Sparse-view 3D Reconstruction in the Wild [[Project](https://jasonyzhang.com/ners/)]
- [[CVPR2022](https://arxiv.org/abs/2110.02624)] CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation [[github](https://github.com/AutodeskAILab/Clip-Forge)]
- [[CVPR2022](https://mworchel.github.io/assets/papers/neural_deferred_shading_with_supp.pdf)] Multi-View Mesh Reconstruction with Neural Deferred Shading [[Project](https://fraunhoferhhi.github.io/neural-deferred-shading/)]
- [[Arxiv](https://arxiv.org/abs/2206.15258)] Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera [[Project](https://ustc3dv.github.io/ndr/)]
- [[Arxiv](https://arxiv.org/abs/2206.08368)] Unbiased 4D: Monocular 4D Reconstruction with a Neural Deformation Model
- [[Arxiv](https://arxiv.org/abs/2205.13914)] 3DILG: Irregular Latent Grids for 3D Generative Modeling [[Project](https://1zb.github.io/3DILG/)]
- [[CVPR2022](https://arxiv.org/abs/2205.07763)] FvOR: Robust Joint Shape and Pose Optimization for Few-view Object Reconstruction [[Project](https://github.com/zhenpeiyang/FvOR/)]
- [[CVPR2022](https://arxiv.org/abs/2205.06267)] Topologically-Aware Deformation Fields for Single-View 3D Reconstruction [[Project](https://shivamduggal4.github.io/tars-3D/)]
- [[Arxiv](https://arxiv.org/abs/2204.10235)] Planes vs. Chairs: Category-guided 3D shape learning without any 3D cues [[Project](https://zixuanh.com/multiclass3D)]
- [[Arxiv](https://arxiv.org/abs/2204.06552)] Neural Vector Fields for Surface Representation and Inference
- [[CVPR2022](https://arxiv.org/abs/2204.03642)] Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction [[Project](https://shubhtuls.github.io/ss3d/)]
- [[CVPR2022](https://arxiv.org/abs/2203.15536)] BARC: Learning to Regress 3D Dog Shape from Images by Exploiting Breed Information [[Project](https://barc.is.tue.mpg.de/)]
- [[CVPR2022](https://arxiv.org/pdf/2203.11938.pdf)] φ-SfT: Shape-from-Template with a Physics-Based Deformation Model [[Project](https://4dqv.mpi-inf.mpg.de/phi-SfT/)]
- [[CVPR2022](https://arxiv.org/abs/2203.07977)] OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction [[Project](https://wenbin-lin.github.io/OcclusionFusion/)]
- [[Arxiv](https://arxiv.org/abs/2202.01999)] Neural Dual Contouring
- [[Arxiv](https://arxiv.org/abs/2201.01831v1)] POCO: Point Convolution for Surface Reconstruction [[Project](https://github.com/valeoai/POCO)]
- [[ICCV2021](https://arxiv.org/abs/2201.00112v1)] SurfGen: Adversarial 3D Shape Synthesis with Explicit Surface Discriminators [[github](https://github.com/aluo-x/NeuralRaycaster)]
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.03258v1)] DoodleFormer: Creative Sketch Drawing with Transformers
- [[NeurIPS2021](https://arxiv.org/abs/2112.02091v1)] Class-agnostic Reconstruction of Dynamic Objects from Videos [[Project](https://jason718.github.io/redo/)]
- [[Arxiv](https://arxiv.org/abs/2112.00584v1)] The Shape Part Slot Machine: Contact-based Reasoning for Generating 3D Shapes from Parts
- [[Arxiv](https://arxiv.org/abs/2111.14549v1)] MeshUDF: Fast and Differentiable Meshing of Unsigned Distance Field Networks [[github](https://github.com/cvlab-epfl/MeshUDF)]
- [[Arxiv](https://arxiv.org/abs/2111.14600v1)] TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers [[github](https://github.com/MegviiRobot/TransMVSNet)]
- [[Arxiv](https://arxiv.org/abs/2111.12772v1)] JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints
- [[Arxiv](https://arxiv.org/abs/2111.11491v1)] Image Based Reconstruction of Liquids from 2D Surface Detections
- [[Arxiv](https://arxiv.org/abs/2201.06845)] TaylorImNet for Fast 3D Shape Reconstruction Based on Implicit Surface Function
- [[NeurIPS2021](https://arxiv.org/abs/2111.04276v1)] Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis [[Project](https://nv-tlabs.github.io/DMTet/)]
- [[ICML2021](https://arxiv.org/pdf/2011.13495.pdf)] Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces [[tensorflow](https://github.com/mabaorui/NeuralPull)]
- [[Arxiv](https://arxiv.org/abs/2112.11427)] StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation [[Project](https://stylesdf.github.io/)]
- [[3DV2021](https://arxiv.org/abs/2110.11599v1)] High Fidelity 3D Reconstructions with Limited Physical Views [[Project](https://sites.google.com/view/high-fidelity-3d-neural-prior)]
- [[3DV2021](https://arxiv.org/abs/2110.11256v1)] Multi-Category Mesh Reconstruction From Image Collections [[github](https://arxiv.org/pdf/2110.11256v1.pdf)]
- [[Arxiv](https://arxiv.org/abs/2110.10784v1)] Style Agnostic 3D Reconstruction via Adversarial Style Transfer [[https://github.com/Felix-Petersen/style-agnostic-3d-reconstruction]()]
- [[Arxiv](https://arxiv.org/abs/2112.12761)] BANMo: Building Animatable 3D Neural Models from Many Casual Videos [[Project](https://banmo-www.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2110.06679v1)] EditVAE: Unsupervised Part-Aware Controllable 3D Point Cloud Shape Generation
- [[Arxiv](https://arxiv.org/abs/2110.05472v1)] Differentiable Stereopsis: Meshes from multiple views using differentiable rendering [[Project](https://shubham-goel.github.io/ds/)]
- [[ICCV2021](https://arxiv.org/abs/2110.03900v1)] Neural Strokes: Stylized Line Drawing of 3D Shapes
- [[ACMMM2021](https://arxiv.org/abs/2109.04153v1)] Single Image 3D Object Estimation with Primitive Graph Networks
- [[Arxiv](https://arxiv.org/abs/2111.12480)] Octree Transformer: Autoregressive 3D Shape Generation on Hierarchically Structured Sequences
- [[Arxiv](https://arxiv.org/abs/2110.06199)] ABO: Dataset and Benchmarks for Real-World 3D Object Understanding [[Project](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)]
- [[ICCV2021](https://arxiv.org/abs/2109.00512)] Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction [[github](https://github.com/facebookresearch/co3d)]
- [[Arxiv](https://arxiv.org/abs/2109.11844)] Learnable Triangulation for Deep Learning-based 3D Reconstruction of Objects of Arbitrary Topology from Single RGB Images
- [[ICCV2021](https://arxiv.org/abs/2108.09964v1)] Learning Signed Distance Field for Multi-view Surface Reconstruction
- [[Arxiv](https://arxiv.org/abs/2108.08477v1)] Image2Lego: Customized LEGO Set Generation from Images
- [[ICCV2021](https://arxiv.org/abs/2108.03746v1)] Unsupervised Learning of Fine Structure Generation for 3D Point Clouds by 2D Projection Matching [[github](https://github.com/chenchao15/2D_projection_matching)]
- [[Arxiv](https://arxiv.org/abs/2108.02708v1)] Object Wake-up: 3-D Object Reconstruction, Animation, and in-situ Rendering from a Single Image
- [[Arxiv](https://arxiv.org/abs/2107.10844v1)] DOVE: Learning Deformable 3D Objects by Watching Videos [[Project](https://dove3d.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2107.09584)] Active 3D Shape Reconstruction from Vision and Touch
- [[NeurIPS2020](https://proceedings.neurips.cc//paper/2020/file/a3842ed7b3d0fe3ac263bcabd2999790-Paper.pdf)] 3D Shape Reconstruction from Vision and Touch [[pytorch](https://github.com/facebookresearch/3D-Vision-and-Touch)]
- [[Arxiv](https://arxiv.org/pdf/2106.12102v1.pdf)] LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2106.08762v1.pdf)] Shape from Blur: Recovering Textured 3D Shape and Motion of Fast Moving Objects
- [[Arxiv](https://arxiv.org/abs/2106.06533v1)] View Generalization for Single Image Textured 3D Models [[Project](https://nv-adlr.github.io/view-generalization)]
- [[Arxiv](https://arxiv.org/pdf/2106.03452v1.pdf)] Shape As Points: A Differentiable Poisson Solver
- [[Arxiv](https://arxiv.org/pdf/2106.03087v1.pdf)] Neural Implicit 3D Shapes from Single Images with Spatial Patterns
- [[IJCAI2021](https://arxiv.org/pdf/2106.01553.pdf)] Spline Positional Encoding for Learning 3D Implicit Signed Distance Fields
- [[Arxiv](https://arxiv.org/pdf/2105.14548.pdf)] Z2P: Instant Rendering of Point Clouds
- [[CVPR2021](https://arxiv.org/abs/2105.11599)] Multi-view 3D Reconstruction of a Texture-less Smooth Surface of Unknown Generic Reflectance
- [[CVPR2021](https://arxiv.org/abs/2105.09396)] Birds of a Feather: Capturing Avian Shape Models from Images [[Project](https://yufu-wang.github.io/aves/)]
- [[Arxiv](https://arxiv.org/pdf/2105.09492.pdf)] DeepCAD: A Deep Generative Network for Computer-Aided Design Models
- [[Arxiv](https://arxiv.org/pdf/2105.08016.pdf)] StrobeNet: Category-Level Multiview Reconstruction of Articulated Objects
- [[CVPR2021](https://arxiv.org/pdf/2105.06663.pdf)] Sketch2Model: View-Aware 3D Modeling from Single Free-Hand Sketches
- [[Arxiv](https://arxiv.org/pdf/2105.03582.pdf)] Sign-Agnostic CONet: Learning Implicit Surface Reconstructions by Sign-Agnostic Optimization of Convolutional Occupancy Networks
- [[IJCAI2021](https://arxiv.org/pdf/2104.14769.pdf)] PointLIE: Locally Invertible Embedding for Point Cloud Sampling and Recovery
- [[Arxiv](https://arxiv.org/abs/2104.10078)] UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction
- [[CVPR2021](https://arxiv.org/pdf/2104.06397.pdf)] Shape and Material Capture at Home
- [[CVPR2021](https://arxiv.org/pdf/2104.05289.pdf)] StereoPIFu: Depth Aware Clothed Human Digitization via Stereo Vision [[Project](https://hy1995.top/StereoPIFuProject/)]
- [[Arxiv](https://arxiv.org/pdf/2104.05652.pdf)] CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly
- [[CVPR2021](https://arxiv.org/pdf/2104.00858v1.pdf)] Fully Understanding Generic Objects:
Modeling, Segmentation, and Reconstruction [[Project](http://cvlab.cse.msu.edu/project-fully3dobject.html)]
- [[CVPR2021](https://arxiv.org/abs/2103.16832v1)] Online Learning of a Probabilistic and Adaptive Scene Representation
- [[CVPR2021](https://arxiv.org/abs/2104.00476)] Fostering Generalization in Single-view 3D Reconstruction by Learning a Hierarchy of Local and Global Shape Priors
- [[Arxiv](https://arxiv.org/abs/2104.00482)] Sketch2Mesh: Reconstructing and Editing 3D Shapes from Sketches
- [[CVPR2021](https://arxiv.org/abs/2103.12266)] Deep Implicit Moving Least-Squares Functions for 3D Reconstruction [[Project](https://github.com/Andy97/DeepMLS)]
- [[Arxiv](https://arxiv.org/pdf/2103.02766.pdf)] PC2WF: 3D WIREFRAME RECONSTRUCTION FROM RAW POINT CLOUDS
- [[CVPR2021](https://arxiv.org/pdf/2103.01458.pdf)] Diffusion Probabilistic Models for 3D Point Cloud Generation
 [[Project](https://github.com/luost26/diffusion-point-cloud)]
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
- [[CVPR2021](https://arxiv.org/pdf/2011.14791.pdf)] NeuralFusion: Online Depth Fusion in Latent Space
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
- [[Arxiv](https://arxiv.org/pdf/2003.03551.pdf)] STD-Net: Structure-preserving and Topology-adaptive Deformation Network for 3D Reconstruction from a Single Image [[new](https://arxiv.org/abs/2108.06682v1)]
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
- [[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yeh_PhotoScene_Photorealistic_Material_and_Lighting_Transfer_for_Indoor_Scenes_CVPR_2022_paper.pdf)] PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes [[github](https://github.com/ViLab-UCSD/photoscene)]
- [[Arxiv](https://arxiv.org/abs/2206.01203)] Semantic Instance Segmentation of 3D Scenes Through Weak Bounding Box Supervision [[Project](http://virtualhumans.mpi-inf.mpg.de/box2mask/)]
- [[CVPR2022](https://arxiv.org/abs/2204.07548)] Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation [[github](https://github.com/drprojects/DeepViewAgg)]
- [[CVPR2022](https://arxiv.org/abs/2204.06272)] 3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection
- [[CVPR2022](https://arxiv.org/abs/2204.06950)] BEHAVE: Dataset and Method for Tracking Human Object Interactions [[Project](http://virtualhumans.mpi-inf.mpg.de/behave/)]
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.11340v1)] Transferable End-to-end Room Layout Estimation via Implicit Encoding [[Project](https://sites.google.com/view/transferrl/)]
- [[Arxiv](https://arxiv.org/abs/2112.10482v1)] ScanQA: 3D Question Answering for Spatial Scene Understanding
- [[Arxiv](https://arxiv.org/abs/2112.08359v1)] 3D Question Answering
- [[Arxiv](https://arxiv.org/abs/2112.06133v1)] MVLayoutNet:3D layout reconstruction with multi-view panoramas
- [[SGP2021](https://arxiv.org/abs/2112.05644v1)] Roominoes: Generating Novel 3D Floor Plans From Existing 3D Rooms
- [[Arxiv](https://arxiv.org/abs/2112.02990v1)] 4DContrast: Contrastive Learning with Dynamic Correspondences for 3D Scene Understanding
- [[Arxiv](https://arxiv.org/abs/2112.03030v1)] Pose2Room: Understanding 3D Scenes from Human Activities [[Project](https://yinyunie.github.io/pose2room-page/)]
- [[NeurIPS2021](https://arxiv.org/abs/2112.01001v1)] SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [[Project](https://devendrachaplot.github.io/projects/seal)]
- [[Arxiv](https://arxiv.org/abs/2112.01551v1)] D3Net: A Speaker-Listener Architecture for Semi-supervised Dense Captioning and Visual Grounding in RGB-D Scans [[Project](https://daveredrum.github.io/D3Net/)]
- [[Arxiv](https://arxiv.org/abs/2112.01520v1)] Recognizing Scenes from Novel Viewpoints
- [[Arxiv](https://arxiv.org/abs/2112.01316v1)] Putting 3D Spatially Sparse Networks on a Diet
- [[Arxiv](https://arxiv.org/abs/2111.12608v1)] Cerberus Transformer: Joint Semantic, Affordance and Attribute Parsing [[github](https://github.com/OPEN-AIR-SUN/Cerberus)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.01253v1)] Neural Scene Flow Prior [[github](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior)]
- [[ICCV2021](https://arxiv.org/abs/2110.01997v1)] Structured Bird's-Eye-View Traffic Scene Understanding from Onboard Images [[Project](https://github.com/ybarancan/STSU)]
- [[Arxiv](https://arxiv.org/abs/2110.00644v1)] RoomStructNet: Learning to Rank Non-Cuboidal Room Layouts From Single View
- [[EMNLP2021](https://arxiv.org/abs/2109.15207v1)] Language-Aligned Waypoint (LAW) Supervision for Vision-and-Language Navigation in Continuous Environments [[Project](https://3dlg-hcvc.github.io/LAW-VLNCE/)]
- [[Arxiv](https://arxiv.org/abs/2109.13410v1)] KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D [[Project](http://www.cvlibs.net/datasets/kitti-360/)]
- [[CVPR2021](https://arxiv.org/abs/2007.12868)] OpenRooms: An End-to-End Open Framework for Photorealistic Indoor Scene Datasets [[github](https://github.com/ViLab-UCSD/OpenRooms)]
- [[Arxiv](https://arxiv.org/abs/2109.08553v1)] Pointly-supervised 3D Scene Parsing with Viewpoint Bottleneck [[github](https://github.com/OPEN-AIR-SUN/Viewpoint-Bottleneck)]
- [[TPAMI2021](https://arxiv.org/abs/2109.05441v1)] Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR-based Perception [[github](https://github.com/xinge008/Cylinder3D)]
- [[Arxiv](https://arxiv.org/abs/2109.05566v1)] PQ-Transformer: Jointly Parsing 3D Objects and Layouts from Point Clouds [[github](https://github.com/OPEN-AIR-SUN/PQ-Transformer)]
- [[Arxiv](https://arxiv.org/abs/2109.04685v1)] Residual 3D Scene Flow Learning with Context-Aware Feature Extraction
- [[ICCV2021](https://arxiv.org/abs/2109.02227v1)] Learning to Generate Scene Graph from Natural Language Supervision [[github](https://github.com/YiwuZhong/SGG_from_NLS)]
- [[ICCV2021](https://arxiv.org/abs/2108.11550v1)] The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation [[Project](https://xiaoming-zhao.github.io/projects/pointnav-vo/)]
- [[ICCV2021](https://arxiv.org/abs/2108.08841v1)] Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs
- [[ICCV2021](https://arxiv.org/abs/2108.06545v1)] PICCOLO: Point Cloud-Centric Omnidirectional Localization
- [[ICCV2021](https://arxiv.org/abs/2108.05884v1)] Unconditional Scene Graph Generation
- [[Arxiv](https://arxiv.org/abs/2108.03378v1)] Learning Indoor Layouts from Simple Point-Clouds
- [[Arxiv](https://arxiv.org/abs/2107.03438v1)] LanguageRefer: Spatial-Language Model for 3D Visual Grounding
- [[Arxiv](https://arxiv.org/abs/2107.01002)] WiCluster: Passive Indoor 2D/3D Positioning using WiFi without Precise Labels
- [[CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cruz_Zillow_Indoor_Dataset_Annotated_Floor_Plans_With_360deg_Panoramas_and_CVPR_2021_paper.html)] Zillow Indoor Dataset: Annotated Floor Plans With 360deg Panoramas and 3D Room Layouts [[github](https://github.com/zillow/zind)]
- [[ICRA2021](https://arxiv.org/abs/2105.09932)] Efficient and Robust LiDAR-Based End-to-End Navigation [[Project](https://le2ed.mit.edu/)]
- [[ICLR2021](https://arxiv.org/pdf/2105.09447.pdf)] VTNet: Visual Transformer Network for Object Goal Navigation
- [[CVPR2021](https://arxiv.org/abs/2105.08248)] Self-Point-Flow: Self-Supervised Scene Flow Estimation from Point Clouds with Optimal Transport and Random Walk
- [[CVPR2021](https://arxiv.org/abs/2105.07751)] HCRF-Flow: Scene Flow from Point Clouds with Continuous High-order CRFs and Position-aware Flow Embedding
- [[Arxiv](https://arxiv.org/pdf/2105.07147.pdf)] FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting
- [[Arxiv](https://arxiv.org/pdf/2105.04447.pdf)] SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation
- [[Arxiv](https://arxiv.org/pdf/2105.01061.pdf)] Collision Replay: What Does Bumping Into Things Tell You About Scene Geometry? [[Project](https://fouheylab.eecs.umich.edu/~alexrais/collisionreplay/)]
- [[Arxiv](https://arxiv.org/abs/2104.11225)] Pri3D: Can 3D Priors Help 2D Representation Learning?
- [[Arxiv](https://arxiv.org/abs/2104.09169)] LaLaLoc: Latent Layout Localisation in Dynamic, Unvisited Environments
- [[CVPRW](https://arxiv.org/abs/2104.09403)] OmniLayout: Room Layout Reconstruction from Indoor Spherical Panoramas [[github](https://github.com/rshivansh/OmniLayout)]
- [[Arxiv](https://arxiv.org/pdf/2104.07986.pdf)] Learning to Reconstruct 3D Non-Cuboid Room Layout from a Single RGB Image [[pytorch](https://github.com/CYang0515/NonCuboidRoom)]
- [[Arxiv](https://arxiv.org/pdf/2104.04891.pdf)] SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000× Fewer Labels [[github](https://github.com/QingyongHu/SQN)]
- [[CVPR2021](https://arxiv.org/abs/2104.00798v1)] FESTA: Flow Estimation via Spatial-Temporal Attention for Scene Point Clouds
- [[CVPR2021](https://arxiv.org/pdf/2103.16381v1.pdf)] Free-form Description Guided 3D Visual Graph Network for Object Grounding in Point Cloud [[github](https://github.com/PNXD/FFL-3DOG)]
- [[ICRA](https://arxiv.org/pdf/2103.16095v1.pdf)] Reconstructing Interactive 3D Scenes by Panoptic Mapping and CAD Model Alignments [[Project](https://sites.google.com/view/icra2021-reconstruction)]
- [[Arxiv](https://arxiv.org/pdf/2103.15369v1.pdf)] Contextual Scene Augmentation and Synthesis via GSACNet
- [[Arxiv](https://arxiv.org/pdf/2103.15875v1.pdf)] In-Place Scene Labelling and Understanding with Implicit Scene Representation
- [[CVPR2021](https://arxiv.org/abs/2103.14326v1)] Bidirectional Projection Network for Cross Dimension Scene Understanding [[github](https://github.com/wbhu/BPNet)]
- [[Arxiv](https://arxiv.org/abs/2103.16381)] Free-form Description Guided 3D Visual Graph Network for Object Grounding in Point Cloud [[github](https://github.com/PNXD/FFL-3DOG)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16544.pdf)] Visual Room Rearrangement [[Project](https://ai2thor.allenai.org/rearrangement/)]
- [[Arxiv](https://arxiv.org/pdf/2103.11161.pdf)] MonteFloor: Extending MCTS for Reconstructing Accurate Large-Scale Floor Plans
- [[Arxiv](https://arxiv.org/pdf/2103.03454.pdf)] Structured Scene Memory for Vision-Language Navigation
- [[Arxiv](https://arxiv.org/pdf/2103.02574.pdf)] House-GAN++: Generative Adversarial Layout Refinement Networks
- [[Arxiv](https://arxiv.org/pdf/2102.08945.pdf)] Weakly Supervised Learning of Rigid 3D Scene Flow
- [[ICLR2021](https://arxiv.org/pdf/2102.07764.pdf)] End-to-End Egospheric Spatial Memory
- [[Arxiv](https://arxiv.org/pdf/2102.03939.pdf)] Single-Shot Cuboids: Geodesics-based End-to-end Manhattan Aligned Layout
Estimation from Spherical Panoramas [[Project](https://vcl3d.github.io/SingleShotCuboids/)]
- [[Arxiv](https://arxiv.org/pdf/2101.07891.pdf)] A modular vision language navigation and manipulation framework for long horizon compositional tasks in indoor environment
- [[Arxiv](https://arxiv.org/pdf/2101.07462.pdf)] Deep Reinforcement Learning for Producing Furniture Layout in Indoor Scenes
- [[Arxiv](https://arxiv.org/pdf/2101.02692.pdf)] Where2Act: From Pixels to Actions for Articulated 3D Objects [[Project](https://cs.stanford.edu/~kaichun/where2act/)]

#### Before 2021
- [[Arxiv](https://arxiv.org/abs/1903.01177)] PanopticFusion: Online Volumetric Semantic Mapping at the Level of Stuff and Things
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
- [[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_PlaneMVS_3D_Plane_Reconstruction_From_Multi-View_Stereo_CVPR_2022_paper.pdf)] PlaneMVS: 3D Plane Reconstruction from Multi-View Stereo
- [[CVPR2022](https://arxiv.org/abs/2205.02836)] Neural 3D Scene Reconstruction with the Manhattan-world Assumption [[Project](https://zju3dv.github.io/manhattan_sdf/)]
- [[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Jeong_3D_Scene_Painting_via_Semantic_Image_Synthesis_CVPR_2022_paper.pdf)] 3D Scene Painting via Semantic Image Synthesis
- [[Siggraph2022](https://arxiv.org/abs/2207.02363)] SNeRF: Stylized Neural Implicit Representations for 3D Scenes [[Project](https://research.facebook.com/publications/snerf-stylized-neural-implicit-representations-for-3d-scenes/)]
- [[Siggraph2022](https://arxiv.org/abs/2205.12955)] Neural 3D Reconstruction in the Wild [[Project](https://zju3dv.github.io/neuralrecon-w/)]
- [[Arxiv](https://arxiv.org/abs/2206.14735)] GO-Surf: Neural Feature Grid Optimization for Fast, High-Fidelity RGB-D Surface Reconstruction [[Project](https://jingwenwang95.github.io/go_surf/)]
- [[Arxiv](https://arxiv.org/abs/2203.13296)] RayTran: 3D pose estimation and shape reconstruction of multiple objects from videos with ray-traced transformers
- [[Arxiv](https://arxiv.org/abs/2204.02296)] iSDF: Real-Time Neural Signed Distance Fields for Robot Perception [[Project](https://joeaortiz.github.io/iSDF/)]
- [[Arxiv](https://arxiv.org/abs/2206.13597)] NeuRIS: Neural Reconstruction of Indoor Scenes Using Normal Priors [[Project](https://jiepengwang.github.io/NeuRIS/)]
- [[CVPR2022](https://arxiv.org/abs/2206.07710)] PlanarRecon: Real-time 3D Plane Detection and Reconstruction from Posed Monocular Videos [[Project](https://neu-vi.github.io/planarrecon/)]
- [[CVPR2022](https://drive.google.com/file/d/1E6xSbUzuu6soAA-jkaGCFl97LZ8SVRvr/view)] Learning 3D Object Shape and Layout without 3D Supervision [[Project](https://gkioxari.github.io/usl/index.html)]
- [[Arxiv](https://arxiv.org/abs/2206.00665)] MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction [[Project](https://niujinshuchong.github.io/monosdf/)]
- [[Arxiv](https://arxiv.org/abs/2205.02837)] BlobGAN: Spatially Disentangled Scene Representations [[Project](https://dave.ml/blobgan/)]
- [[CVPR2022](https://arxiv.org/abs/2203.11283)] NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction
- [[Arxiv](https://arxiv.org/abs/2202.00185v1)] ATEK: Augmenting Transformers with Expert Knowledge for Indoor Layout Synthesis
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.05126v1)] IterMVS: Iterative Probability Estimation for Efficient Multi-View Stereo [[github](https://github.com/FangjinhuaWang/IterMVS)]
- [[Arxiv](https://arxiv.org/abs/2112.04481v1)] What's Behind the Couch? Directed Ray Distance Functions (DRDF) for 3D Scene Reconstruction [[Project](https://nileshkulkarni.github.io/scene_drdf/)]
- [[Arxiv](https://arxiv.org/abs/2112.03243v1)] Input-level Inductive Biases for 3D Reconstruction
- [[Arxiv](https://arxiv.org/abs/2112.01988v1)] ROCA: Robust CAD Model Retrieval and Alignment from a Single Image
- [[Arxiv](https://arxiv.org/abs/2112.00336v1)] Multi-View Stereo with Transformer
- [[3DV2021](https://arxiv.org/abs/2112.00202v1)] 3DVNet: Multi-View Depth Prediction and Volumetric Refinement
- [[Arxiv](https://arxiv.org/abs/2112.00236v1)] VoRTX: Volumetric 3D Reconstruction With Transformers for Voxelwise View Selection and Fusion
- [[Arxiv](https://arxiv.org/abs/2111.12905v1)] CIRCLE: Convolutional Implicit Reconstruction and Completion for Large-scale Indoor Scene
- [[Arxiv](https://arxiv.org/abs/2111.12924v1)] Joint stereo 3D object detection and implicit surface reconstruction
- [[CoRL2021](https://arxiv.org/abs/2111.07418v1)] TANDEM: Tracking and Dense Mapping in Real-time using Deep Multi-view Stereo [[Project](https://vision.in.tum.de/research/vslam/tandem)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.03098v1)] Voxel-based 3D Detection and Reconstruction of Multiple Objects from a Single Image [[Project](http://cvlab.cse.msu.edu/project-mdr.html)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.02444v1)] Panoptic 3D Scene Reconstruction From a Single RGB Image
- [[Arxiv](https://arxiv.org/abs/2112.12130)] NICE-SLAM: Neural Implicit Scalable Encoding for SLAM [[Project](https://pengsongyou.github.io/nice-slam)]
- [[BMVC2021](https://arxiv.org/abs/2110.11219v1)] PlaneRecNet: Multi-Task Learning with Cross-Task Consistency for Piece-Wise Plane Detection and Reconstruction from a Single RGB Image [[github](https://github.com/EryiXie/PlaneRecNet)]
- [[ICCV2021](https://arxiv.org/abs/2108.13499)] Scene Synthesis via Uncertainty-Driven Attribute Synchronization [[github](https://github.com/yanghtr/Sync2Gen)]
- [[NeurIPS2021](https://arxiv.org/abs/2110.03675)] ATISS: Autoregressive Transformers for Indoor Scene Synthesis [[Project](https://nv-tlabs.github.io/ATISS/)]
- [[ICCV2021](https://arxiv.org/abs/2109.06061)] Learning Indoor Inverse Rendering with 3D Spatially-Varying Lighting
- [[Arxiv](https://arxiv.org/abs/2108.09911v1)] Black-Box Test-Time Shape REFINEment for Single View 3D Reconstruction
- [[Arxiv](https://arxiv.org/abs/2108.09022v1)] Indoor Scene Generation from a Collection of Semantic-Segmented Depth Images
- [[ICCV2021](https://arxiv.org/abs/2108.08378v1)] Vis2Mesh: Efficient Mesh Reconstruction from Unstructured Point Clouds of Large Scenes with Learned Virtual View Visibility [[github](https://github.com/GDAOSU/vis2mesh)]
- [[ICCV2021](https://arxiv.org/abs/2108.08653v1)] 3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces [[Project](https://myavartanoo.github.io/3dias/)]
- [[ICCV2021](https://arxiv.org/abs/2108.08623v1)] VolumeFusion: Deep Depth Fusion for 3D Scene Reconstruction
- [[Arxiv](https://arxiv.org/abs/2108.03824v1)] AA-RMVSNet: Adaptive Aggregation Recurrent Multi-view Stereo Network
- [[Arxiv](https://arxiv.org/abs/2108.03880v1)] NeuralMVS: Bridging Multi-View Stereo and Novel View Synthesis
- [[ICCV2021](https://arxiv.org/abs/2107.14790v1)] Out-of-Core Surface Reconstruction via Global $TGV$ Minimization
- [[ICCV2021](https://arxiv.org/abs/2107.13629v1)] Discovering 3D Parts from Image Collections [[Project](https://chhankyao.github.io/lpd/)]
- [[ICCV2021](https://arxiv.org/abs/2107.13108v1)] PlaneTR: Structure-Guided Transformers for 3D Plane Recovery [[pytorch](https://github.com/IceTTTb/PlaneTR3D)]
- [[Arxiv](https://arxiv.org/abs/2107.02191)] TransformerFusion: Monocular RGB Scene Reconstruction using Transformers [[Project](https://aljazbozic.github.io/transformerfusion/)]
- [[Arxiv](https://arxiv.org/pdf/2106.14166v1.pdf)] Indoor Panorama Planar 3D Reconstruction via Divide and Conquer
- [[Arxiv](https://arxiv.org/pdf/2106.10689v1.pdf)] NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction
- [[CVPR2021](https://arxiv.org/pdf/2106.06629v1.pdf)] Mirror3D: Depth Refinement for Mirror Surfaces [[Project](https://3dlg-hcvc.github.io/mirror3d/#/)]
- [[CVPR2021](https://arxiv.org/abs/2106.05375v1)] Plan2Scene: Converting Floorplans to 3D Scenes [[Project](https://3dlg-hcvc.github.io/plan2scene/)]
- [[Arxiv](https://arxiv.org/abs/2106.00912)] Translational Symmetry-Aware Facade Parsing for 3D Building Reconstruction
- [[Arxiv](https://arxiv.org/abs/2105.13509)] Learning to Stylize Novel Views [[Project](https://hhsinping.github.io/3d_scene_stylization/)]
- [[Arxiv](https://arxiv.org/pdf/2105.13016.pdf)] Stylizing 3D Scene via Implicit Representation and HyperNetwork
- [[CVPR2021](https://arxiv.org/abs/2105.08612)] SAIL-VOS 3D: A Synthetic Dataset and Baselines for Object Detection and 3D Mesh Reconstruction from Video Data [[Project](http://sailvos.web.illinois.edu/_site/index.html)]
- [[Arxiv](https://arxiv.org/abs/2105.08052)] The Boombox: Visual Reconstruction from Acoustic Vibrations [[Project](https://boombox.cs.columbia.edu/)]
- [[Arxiv](https://arxiv.org/pdf/2009.03964.pdf)] Joint Pose and Shape Estimation of Vehicles from LiDAR Data
- [[CVPR2021](https://arxiv.org/pdf/2104.00681v1.pdf)] NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video [[Project](https://zju3dv.github.io/neuralrecon/)]
- [[Arxiv](https://arxiv.org/abs/2103.14275v1)] DDR-Net: Learning Multi-Stage Multi-View Stereo With Dynamic Depth Range [[pytorch](https://github.com/Tangshengku/DDR-Net)]
- [[Arxiv](https://arxiv.org/pdf/2103.14644v1.pdf)] Planar Surface Reconstruction from Sparse Views [[Project](https://jinlinyi.github.io/SparsePlanes/)]
- [[Arxiv](https://arxiv.org/pdf/2104.04532.pdf)] Neural RGB-D Surface Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2104.00024.pdf)] RetrievalFuse: Neural 3D Scene Reconstruction with a Database
- [[ICCV2021](https://arxiv.org/abs/2103.14024)] PlenOctrees for Real-time Rendering of Neural Radiance Fields [[C++](https://github.com/sxyu/volrend)]
- [[Arxiv](https://arxiv.org/abs/2103.12352)] iMAP: Implicit Mapping and Positioning in Real-Time
- [[CVPR2021](https://arxiv.org/pdf/2103.07969.pdf)] Monte Carlo Scene Search for 3D Scene Understanding
- [[CVPR2021](https://arxiv.org/abs/2103.06422v1)] Holistic 3D Scene Understanding from a Single Image with Implicit Representation
- [[CVPR2021](https://arxiv.org/pdf/2011.14744.pdf)] RfD-Net: Point Scene Understanding by Semantic Instance Reconstruction [[pytorch](https://github.com/yinyunie/RfDNet)]
- [[Arxiv](https://arxiv.org/pdf/2102.13090.pdf)] IBRNet: Learning Multi-View Image-Based Rendering [[Project](https://ibrnet.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2012.02190.pdf)] STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in Motion with Neural Rendering [[Project](https://wentaoyuan.github.io/star/)]

#### Before 2021
- [[ToG2018](https://dl.acm.org/doi/10.1145/3197517.3201362)] Deep convolutional priors for indoor scene synthesis [[github](https://github.com/brownvc/deep-synth)]
- [[Arxiv](https://arxiv.org/pdf/2012.05360.pdf)] MO-LTR: Multiple Object Localization, Tracking and Reconstruction from Monocular RGB Videos
- [[Arxiv](https://arxiv.org/pdf/2012.05551.pdf)] DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors
- [[3DV2020](https://arxiv.org/pdf/2011.00320.pdf)] Scene Flow from Point Clouds with or without Learning
- [[Arxiv](https://arxiv.org/abs/2011.07233)] Stable View Synthesis
- [[Arxiv](https://arxiv.org/pdf/2011.10379.pdf)] Neural Scene Graphs for Dynamic Scenes
- [[3DV2020](https://arxiv.org/pdf/2011.10359.pdf)] RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty [[pytorch](https://github.com/facebookresearch/RidgeSfM)]
- [[Arxiv](https://arxiv.org/pdf/2011.10147.pdf)] FlowStep3D: Model Unrolling for Self-Supervised Scene Flow Estimation
- [[Arxiv](https://arxiv.org/pdf/2011.10812.pdf)] MoNet: Motion-based Point Cloud Prediction Network
- [[Arxiv](https://arxiv.org/pdf/2011.11814.pdf)] MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments from a Single Moving Camera
- [[Arxiv](https://arxiv.org/pdf/2011.11986.pdf)] Efficient Initial Pose-graph Generation for Global SfM
- [[Arxiv](https://arxiv.org/abs/2011.13084)] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes [[Project](http://www.cs.cornell.edu/~zl548/NSFF/)]
- [[Arxiv](https://arxiv.org/pdf/2011.14398.pdf)] RGBD-Net: Predicting color and depth images for novel views synthesis
- [[Arxiv](https://arxiv.org/pdf/2012.04512.pdf)] SSCNav: Confidence-Aware Semantic Scene Completion for Visual Semantic Navigation [[Project](https://sscnav.cs.columbia.edu/)]
- [[Arxiv](https://arxiv.org/pdf/2012.11575.pdf)] From Points to Multi-Object 3D Reconstruction
- [[Arxiv](https://worldsheet.github.io/resources/worldsheet.pdf)] Worldsheet: Wrapping the World in a 3D Sheet
for View Synthesis from a Single Image [[Project](https://worldsheet.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2012.09793.pdf)] SceneFormer: Indoor Scene Generation with Transformers [[pytorch](https://github.com/cy94/sceneformer)]
- [[NeurIPS2020](https://proceedings.neurips.cc/paper/2020/file/b4b758962f17808746e9bb832a6fa4b8-Paper.pdf)] Neural Sparse Voxel Fields [[Project](https://lingjie0206.github.io/papers/NSVF/)]
- [[Arxiv](https://arxiv.org/pdf/2012.02094.pdf?fbclid=IwAR03XwEdhXUl2lsLr20dOnFEsnthPBbdVi9VHDni6CYnhH9glzGaooU-DHM)] Towards Part-Based Understanding of RGB-D Scans
- [[Arxiv](https://arxiv.org/pdf/2011.05813.pdf)] Dynamic Plane Convolutional Occupancy Networks
- [[NeurIPS2020](https://arxiv.org/pdf/2010.13938.pdf)] Neural Unsigned Distance Fields for Implicit Function Learning [[Project](http://virtualhumans.mpi-inf.mpg.de/ndf/)]
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
## NeRF
- [[Arxiv](https://arxiv.org/abs/2207.01583)] LaTeRF: Label and Text Driven Object Radiance Fields
- [[Arxiv](https://arxiv.org/abs/2206.04669)] Beyond RGB: Scene-Property Synthesis with Neural Radiance Fields
- [[CVPR2022](https://arxiv.org/abs/2206.06481)] RigNeRF: Fully Controllable Neural 3D Portraits [[Project](http://shahrukhathar.github.io/2022/06/06/RigNeRF.html)]
- [[Arxiv](https://arxiv.org/abs/2205.04334)] Panoptic Neural Fields: A Semantic Object-Aware Neural Scene Representation
- [[Arxiv](https://d2nerf.github.io/D%5E2NeRF%20Self-Supervised%20Decoupling%20of%20Dynamic%20and%20Static%20Objects%20from%20a%20Monocular%20Video.pdf)] D2NeRF: Self-Supervised Decoupling of Dynamic and Static Objects from a Monocular Video [[Project](https://d2nerf.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2202.05628)] Artemis: Articulated Neural Pets with Appearance and Motion synthesis [[Project](https://haiminluo.github.io/publication/artemis/)]
- [[Arxiv](https://arxiv.org/abs/2205.04992)] KeypointNeRF: Generalizing Image-based Volumetric Avatars using Relative Spatial Encoding of Keypoints [[Project](https://markomih.github.io/KeypointNeRF/)]
- [[Arxiv](https://arxiv.org/abs/2204.10850)] Control-NeRF: Editable Feature Volumes for Scene Rendering and Manipulation
- [[Arxiv](https://arxiv.org/abs/2202.04879v1)] PVSeRF: Joint Pixel-, Voxel- and Surface-Aligned Radiance Field for Single-Image Novel View Synthesis
- [[Arxiv](https://arxiv.org/abs/2202.05263v1)] Block-NeRF: Scalable Large Scene Neural View Synthesis [[Project](https://waymo.com/intl/zh-cn/research/block-nerf/)]
- [[Arxiv](https://arxiv.org/abs/2202.13162)] Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation
- [[Arxiv](https://arxiv.org/abs/2111.13260)] NeSF: Neural Semantic Fields for Generalizable Semantic Segmentation of 3D Scenes [[Project](https://nesf3d.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2201.04127v1)] HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video [[github](https://grail.cs.washington.edu/projects/humannerf/)]
- [[Arxiv](https://arxiv.org/abs/2201.02533v1)] NeROIC: Neural Rendering of Objects from Online Image Collections [[Projetc](https://formyfamily.github.io/NeROIC/)]
- [[Arxiv](https://arxiv.org/abs/2201.00791v1)] DFA-NeRF: Personalized Talking Head Generation via Disentangled Face Attributes Neural Rendering
- [[Arxiv](https://arxiv.org/abs/2112.15399v1)] InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering [[Project](http://cvlab.snu.ac.kr/research/InfoNeRF)]
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.10703v1)] Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs [[Project](https://meganerf.cmusatyalab.org/)]
- [[Arxiv](https://arxiv.org/abs/2112.09687v1)] Light Field Neural Rendering [[Project](https://light-field-neural-rendering.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2112.03517v1)] CG-NeRF: Conditional Generative Neural Radiance Fields
- [[Arxiv](https://arxiv.org/abs/2112.03907v1)] Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields [[Project](https://dorverbin.github.io/refnerf/)]
- [[Arxiv](https://arxiv.org/abs/2112.02308v1)] MoFaNeRF: Morphable Facial Neural Radiance Field
- [[Arxiv](https://arxiv.org/abs/2112.03288v1)] Dense Depth Priors for Neural Radiance Fields from Sparse Input Views
- [[Arxiv](https://arxiv.org/abs/2112.01759v1)] NeRF-SR: High-Quality Neural Radiance Fields using Super-Sampling [[Project](https://cwchenwang.github.io/nerf-sr/)]
- [[Arxiv](https://arxiv.org/abs/2112.00724v1)] RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs [[Project](https://m-niemeyer.github.io/regnerf/index.html)]
- [[Arxiv](https://arxiv.org/abs/2111.15234v1)] NeRFReN: Neural Radiance Fields with Reflections [[Project](https://bennyguo.github.io/nerfren/)]
- [[Arxiv](https://arxiv.org/abs/2111.15552v1)] NeuSample: Neural Sample Field for Efficient View Synthesis [[Project](https://jaminfong.cn/neusample/)]
- [[Arxiv](https://arxiv.org/abs/2111.14643v1)] Urban Radiance Fields [[Project](https://urban-radiance-fields.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2111.13539v1)] GeoNeRF: Generalizing NeRF with Geometry Priors [[Project](https://www.idiap.ch/paper/geonerf/)]
- [[Arxiv](https://arxiv.org/abs/2111.13679v1)] NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images [[Project](https://bmild.github.io/rawnerf/)]
- [[Arxiv](https://arxiv.org/abs/2111.13112v1)] VaxNeRF: Revisiting the Classic for Voxel-Accelerated Neural Radiance Field [[github](https://github.com/naruya/VaxNeRF)]
- [[Arxiv](https://arxiv.org/abs/2111.11215v1)] Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction [[github](https://github.com/sunset1995/DirectVoxGO)]
- [[Arxiv](https://arxiv.org/abs/2111.09996v1)] LOLNeRF: Learn from One Look
- [[Arxiv](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding [[Project](https://nvlabs.github.io/instant-ngp/)]
- [[NeurIPS2021](https://arxiv.org/abs/2110.14213v1)] Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose [[github](https://github.com/Angtian/NeuralVS)]
- [[Arxiv](https://arxiv.org/abs/2112.05598)] PERF: Performant, Explicit Radiance Fields
- [[Arxiv](https://arxiv.org/abs/2112.05131)] Plenoxels: Radiance Fields without Neural Networks [[Project](https://alexyu.net/plenoxels/)]
- [[NeurIPS2021](https://arxiv.org/abs/2109.07448v1)] Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering [[Project](https://youngjoongunc.github.io/nhp/)]
- [[ICCV2021](https://arxiv.org/abs/2109.01750v1)] CodeNeRF: Disentangled Neural Radiance Fields for Object Categories [[github](https://github.com/wayne1123/code-nerf)]
- [[ICCV2021](https://arxiv.org/abs/2109.01847v1)] Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering [[Project](https://zju3dv.github.io/object_nerf/)]
- [[ICCV2021](https://arxiv.org/abs/2108.04886v1)] Differentiable Surface Rendering via Non-Differentiable Sampling
- [[ICCV2021](https://arxiv.org/abs/2104.00677)] Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis [[Project](https://www.ajayj.com/dietnerf)]
- [[Arxiv](https://arxiv.org/abs/2107.05775v1)] Fast and Explicit Neural View Synthesis
- [[Arxiv](https://arxiv.org/abs/2107.02791v1)] Depth-supervised NeRF: Fewer Views and Faster Training for Free [[Project](https://www.cs.cmu.edu/~dsnerf/)] [[pytorch](https://github.com/dunbar12138/DSNeRF)]
- [[Arxiv](https://arxiv.org/pdf/2106.13228.pdf)] A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields [[Project](https://hypernerf.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2106.05264v1)] NeRF in detail: Learning to sample for view synthesis
- [[Arxiv](https://arxiv.org/pdf/2106.01970.pdf)] NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination [[Project](https://people.csail.mit.edu/xiuming/projects/nerfactor/)]
- [[Arxiv](https://arxiv.org/abs/2105.05994)] Neural Trajectory Fields for Dynamic Novel View Synthesis
- [[Arxiv](http://editnerf.csail.mit.edu/paper.pdf)] Editing Conditional Radiance Fields [[Project](http://editnerf.csail.mit.edu/)]
- [[CVPR2021](https://arxiv.org/abs/2104.06935)] Stereo Radiance Fields (SRF): Learning View Synthesis for Sparse Views of Novel Scenes
- [[Arxiv](https://arxiv.org/pdf/2103.15606v2.pdf)] GNeRF: GAN-based Neural Radiance Field without Posed Camera 
- [[Arxiv](https://arxiv.org/pdf/2104.06405.pdf)] BARF: Bundle-Adjusting Neural Radiance Fields [[Project](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/)]
- [[Arxiv](https://arxiv.org/pdf/2103.15595v1.pdf)] MVSNeRF: Fast Generalizable Radiance Field Reconstruction
from Multi-View Stereo
- [[CVPR2021](https://arxiv.org/abs/2103.11571)] Neural Lumigraph Rendering [[Project](http://www.computationalimaging.org/publications/nlr/)]
- [[Arxiv](https://arxiv.org/abs/2103.13415)] Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- [[Arxiv](https://arxiv.org/abs/2103.13744)] KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs
- [[Arxiv](https://arxiv.org/pdf/2103.10380.pdf)] FastNeRF: High-Fidelity Neural Rendering at 200FPS
- [[CVPR2021](https://arxiv.org/pdf/2103.05606.pdf)] NeX: Real-time View Synthesis with Neural Basis Expansion [[Project](https://nex-mpi.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2103.03231)] DONeRF: Towards Real-Time Rendering of Neural Radiance Fields using Depth Oracle Networks [[Project](https://depthoraclenerf.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2102.07064)] NeRF--: Neural Radiance Fields Without Known Camera Parameters [[Project](http://nerfmm.active.vision/)]

#### Before 2021
- [[Arxiv](https://arxiv.org/pdf/2012.02190.pdf)] pixelNeRF: Neural Radiance Fields from One or Few Images [[Project](https://alexyu.net/pixelnerf/)]
- [[Arxiv](https://arxiv.org/pdf/2012.03927.pdf)] NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis [[Project](https://people.eecs.berkeley.edu/~pratul/nerv/)]
- [[Arxiv](https://arxiv.org/pdf/2012.09790.pdf)] Neural Radiance Flow for 4D View Synthesis and Video Processing [[Project](https://yilundu.github.io/nerflow/)]
- [[Arxiv](https://arxiv.org/abs/2011.12948)] Deformable Neural Radiance Fields [[Project](https://nerfies.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2011.12490.pdf)] DeRF: Decomposed Radiance Fields
- [[Arxiv](https://arxiv.org/pdf/2003.08934.pdf)] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis











---
## About Human Body
- [[ECCV2022](https://arxiv.org/abs/2207.12824)] Compositional Human-Scene Interaction Synthesis with Semantic Control [[Project](https://github.com/zkf1997/COINS)]
- [[ECCV2022](https://arxiv.org/abs/2207.11770)] Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis [[Project](https://sstzal.github.io/DFRF/)]
- [[CVPR2022](https://arxiv.org/abs/2204.08906)] Photorealistic Monocular 3D Reconstruction of Humans Wearing Clothing [[Project](https://www.liuyebin.com/diffustereo/diffustereo.html)]
- [[ECCV2022](https://arxiv.org/abs/2207.08000)] DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras [[Project](https://www.liuyebin.com/diffustereo/diffustereo.html)]
- [[CVPR2022](https://arxiv.org/abs/2206.09553)] Capturing and Inferring Dense Full-Body Human-Scene Contact [[Project](https://rich.is.tue.mpg.de/)]
- [[Arxiv](https://arxiv.org/abs/2206.08343)] Realistic One-shot Mesh-based Head Avatars [[Project](https://samsunglabs.github.io/rome/)]
- [[CVPR2022](https://arxiv.org/abs/2204.10211)] SmartPortraits: Depth Powered Handheld Smartphone Dataset of Human Portraits for State Estimation, Reconstruction and Synthesis
- [[Arxiv](https://arxiv.org/abs/2204.03688)] DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[Project](https://www.pinatafarm.com/research/dad-3dheads)]
- [[CVPR2022](https://arxiv.org/abs/2203.14478)] Structured Local Radiance Fields for Human Avatar Modeling
- [[CVPR2022](https://arxiv.org/abs/2203.14510)] ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations
- [[Arxiv](https://arxiv.org/abs/2203.13817)] AutoAvatar: Autoregressive Neural Fields for Dynamic Avatar Modeling [[Project](https://zqbai-jeremy.github.io/autoavatar/)]
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.09251v1)] The Wanderings of Odysseus in 3D Scenes [[Project](https://yz-cnsdqz.github.io/eigenmotion/GAMMA/)]
- [[Arxiv](https://arxiv.org/abs/2112.08274v1)] Putting People in their Place: Monocular Regression of 3D People in Depth [[github](https://github.com/Arthur151/ROMP)]
- [[Arxiv](https://arxiv.org/abs/2112.04477v1)] Tracking People by Predicting 3D Appearance, Location &amp; Pose [[Project](http://people.eecs.berkeley.edu/~jathushan/PHALP/)]
- [[Arxiv](https://arxiv.org/abs/2112.04203v1)] Adversarial Parametric Pose Prior
- [[NeurIPS2021](https://arxiv.org/abs/2112.04159v1)] Garment4D: Garment Reconstruction from Point Cloud Sequences [[Project](https://hongfz16.github.io/projects/Garment4D.html)]
- [[Arxiv](https://arxiv.org/abs/2112.02753v1)] MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image [[github](https://github.com/SeanChenxy/HandMesh)]
- [[Arxiv](https://arxiv.org/abs/2112.02082v1)] Total Scale: Face-to-Body Detail Reconstruction from Sparse RGBD Sensors
- [[Arxiv](https://arxiv.org/abs/2112.01524v1)] GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras [[Project](https://www.ye-yuan.com/glamr/)]
- [[3DV2021](https://arxiv.org/abs/2111.15113v1)] LatentHuman: Shape-and-Pose Disentangled Latent Representation for Human Bodies [[Project](https://latenthuman.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2111.12696v1)] A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose
- [[Arxiv](https://arxiv.org/abs/2111.12707v1)] MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation [[github](https://github.com/Vegetebird/MHFormer)]
- [[Arxiv](https://arxiv.org/abs/2111.12073v1)] Multi-Person 3D Motion Prediction with Multi-Range Transformers [[Project](https://jiashunwang.github.io/MRT/)]
- [[Arxiv](https://arxiv.org/abs/2112.12390)] DD-NeRF: Double-Diffusion Neural Radiance Field as a Generalizable Implicit Body Representation
- [[Arxiv](https://arxiv.org/abs/2110.11746v1)] Creating and Reenacting Controllable 3D Humans with Differentiable Rendering
- [[Arxiv](https://arxiv.org/abs/2110.11680v1)] Deep Two-Stream Video Inference for Human Body Pose and Shape Estimation
- [[BMVC2021](https://arxiv.org/abs/2110.10533v1)] AniFormer: Data-driven 3D Animation with Transformer [[Project](https://github.com/mikecheninoulu/AniFormer)]
- [[ACMMM2021](https://arxiv.org/abs/2110.08729v1)] VoteHMR: Occlusion-Aware Voting Network for Robust 3D Human Mesh Recovery from Partial Point Clouds
- [[Arxiv](https://arxiv.org/abs/2110.07588v1)] Playing for 3D Human Recovery [[Project](https://gta-human.com/)]
- [[ICCV2021](https://arxiv.org/abs/2110.03480v1)] Learning to Regress Bodies from Images using Differentiable Semantic Rendering [[Project](https://dsr.is.tue.mpg.de/)]
- [[Arxiv](https://arxiv.org/abs/2112.09127)] ICON: Implicit Clothed humans Obtained from Normals [[github](https://github.com/YuliangXiu/ICON)]
- [[ICCV2021](https://arxiv.org/abs/2110.00990v1)] Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild [[Project](https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman)]
- [[Arxiv](https://arxiv.org/abs/2110.00620v1)] SPEC: Seeing People in the Wild with an Estimated Camera [[Project](https://spec.is.tue.mpg.de/)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.07868)] Tracking People with 3D Representations [[github](https://github.com/brjathu/T3DP)]
- [[Arxiv](https://arxiv.org/abs/2109.11399v1)] A Skeleton-Driven Neural Occupancy Representation for Articulated Hands
- [[Arxiv](https://arxiv.org/abs/2109.08364v1)] GraFormer: Graph Convolution Transformer for 3D Pose Estimation [[github](https://github.com/Graformer/GraFormer)]
- [[ICCV2021](https://arxiv.org/abs/2109.05885v1)] Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images
- [[ICCV2021](https://arxiv.org/abs/2109.02303v1)] Encoder-decoder with Multi-level Attention for 3D Human Shape and Pose Estimation [[github](https://github.com/ziniuwan/maed)]
- [[ICCV2021](https://arxiv.org/abs/2109.02563v1)] 3D Human Texture Estimation from a Single Image with Transformers
- [[ICCV2021](https://arxiv.org/abs/2109.00033v1)] DensePose 3D: Lifting Canonical Surface Maps of Articulated Objects to the Third Dimension
- [[Arxiv](https://arxiv.org/abs/2104.03953)] SNARF: Differentiable Forward Skinning for Animating Non-Rigid Neural Implicit Shapes [[Project](https://xuchen-ethz.github.io/snarf/)]
- [[ICCV2021](https://arxiv.org/abs/2108.11944v1)] Probabilistic Modeling for Human Mesh Recovery [[Project](https://www.seas.upenn.edu/~nkolot/projects/prohmr/)]
- [[ICCV2021](https://arxiv.org/abs/2108.11609v1)] Unsupervised Dense Deformation Embedding Network for Template-Free Shape Correspondence
- [[ACMMM2021](https://arxiv.org/abs/2108.12384v1)] DC-GNet: Deep Mesh Relation Capturing Graph Convolution Network for 3D Human Shape Reconstruction
- [[SiggraphAsia2019](https://dl.acm.org/doi/pdf/10.1145/3355089.3356505)] Neural State Machine for Character-Scene Interactions [[github](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_Asia_2019)]
- [[ICCV2021](https://arxiv.org/abs/2108.10399v1)] Learning Motion Priors for 4D Human Body Capture in 3D Scenes [[Project](https://sanweiliti.github.io/LEMO/LEMO.html)]
- [[Arxiv](https://arxiv.org/abs/2108.09000v1)] Deep Virtual Markers for Articulated 3D Shapes
- [[ICCV2021](https://arxiv.org/abs/2108.08844v1)] Gravity-Aware Monocular 3D Human-Object Reconstruction [[Project](http://4dqv.mpi-inf.mpg.de/GraviCap/)]
- [[ICCV2021](https://arxiv.org/abs/2108.08478v1)] Learning Anchored Unsigned Distance Functions with Gradient Direction Alignment for Single-view Garment Reconstruction
- [[Arxiv](https://arxiv.org/abs/2108.08420v1)] D3D-HOI: Dynamic 3D Human-Object Interactions from Videos [[github](https://github.com/facebookresearch/d3d-hoi)]
- [[ICCV2021](https://arxiv.org/abs/2108.08284v1)] Stochastic Scene-Aware Motion Prediction [[Project](https://samp.is.tue.mpg.de/)] [[github](https://github.com/mohamedhassanmus/SAMP)]
- [[ICCV2021](https://arxiv.org/abs/2108.07845v1)] ARCH++: Animation-Ready Clothed Human Reconstruction Revisited
- [[ICCV2021](https://arxiv.org/abs/2108.06819v1)] EventHPE: Event-based 3D Human Pose and Shape Estimation
- [[ACMMM2021](https://arxiv.org/abs/2108.04536v1)] Learning Multi-Granular Spatio-Temporal Graph Network for Skeleton-based Action Recognition [[github](https://github.com/tailin1009/DualHead-Network)]
- [[ACMMM2021](https://arxiv.org/abs/2108.03656v1)] Skeleton-Contrastive 3D Action Representation Learning [[github](https://github.com/fmthoker/skeleton-contrast)]
- [[Arxiv](https://arxiv.org/abs/2107.12847v1)] Learning Local Recurrent Models for Human Mesh Recovery
- [[Arxiv](https://arxiv.org/abs/2107.12512)] H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction [[Project](https://crisalixsa.github.io/h3d-net/)]
- [[Arxiv](https://arxiv.org/abs/2107.07539v1)] Unsupervised 3D Human Mesh Recovery from Noisy Point Clouds [[github](https://github.com/wangsen1312/unsupervised3dhuman)]
- [[Arxiv](https://arxiv.org/pdf/2106.11944v1.pdf)] MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images [[Project](https://neuralbodies.github.io/metavatar/)]
- [[Arxiv](https://arxiv.org/pdf/2106.11536v1.pdf)] Deep3DPose: Realtime Reconstruction of Arbitrarily Posed Human Bodies from Single RGB Images
- [[Arxiv](https://arxiv.org/pdf/2106.09336v1.pdf)] THUNDR: Transformer-based 3D HUmaN Reconstruction with Markers
- [[CVPR2021](http://www.liuyebin.com/Function4D/assets/Function4D.pdf)] Function4D: Real-time Human Volumetric Capture from Very Sparse RGBD Sensors [[Project](http://www.liuyebin.com/Function4D/Function4D.html)]
- [[Arxiv](https://arxiv.org/abs/2106.06313v1)] Bridge the Gap Between Model-based and Model-free Human Reconstruction
- [[Arxiv](https://arxiv.org/pdf/2106.02019.pdf)] Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control
- [[Arxiv](https://arxiv.org/pdf/2105.14804.pdf)] Scene-aware Generative Network for Human Motion Synthesis
- [[Arxiv](https://arxiv.org/pdf/2105.08715.pdf)] Human Motion Prediction Using Manifold-Aware Wasserstein GAN
- [[CVPR2021](https://arxiv.org/abs/2105.01859)] Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors [[Project](http://www.liuyebin.com/Function4D/Function4D.html)]
- [[Arxiv](https://arxiv.org/pdf/2104.04029.pdf)] TRiPOD: Human Trajectory and Pose Dynamics Forecasting in the Wild [[Project](http://somof.stanford.edu/results)]
- [[CVPR2021](https://arxiv.org/pdf/2012.00619.pdf)] We are More than Our Joints: Predicting how 3D Bodies Move [[Project](https://yz-cnsdqz.github.io/MOJO/MOJO.html)]
- [[CVPR2021](https://arxiv.org/pdf/2104.06849.pdf)] LEAP: Learning Articulated Occupancy of People [[Project](https://neuralbodies.github.io/LEAP/)]
- [[Arxiv](https://arxiv.org/pdf/2104.07300.pdf)] 3DCrowdNet: 2D Human Pose-Guided 3D Crowd Human Pose and Shape Estimation in the Wild
- [[CVPR2021](https://arxiv.org/abs/2104.07660)] SCALE: Modeling Clothed Humans with a Surface Codec of Articulated Local Elements [[Project](https://qianlim.github.io/SCALE)]
- [[Arxiv](https://arxiv.org/pdf/2104.05670.pdf)] Action-Conditioned 3D Human Motion Synthesis with Transformer VAE
 [[Project](https://imagine.enpc.fr/~petrovim/actor/)]
- [[Arxiv](https://arxiv.org/pdf/2104.03978.pdf)] Dynamic Surface Function Networks for Clothed Human Bodies [[github](https://github.com/andreiburov/DSFN)]
- [[Arxiv](https://arxiv.org/abs/2104.03110v1)] Neural Articulated Radiance Field [[github](https://github.com/nogu-atsu/NARF)]
- [[Arxiv](https://arxiv.org/pdf/2104.00272v1.pdf)] Mesh Graphormer
- [[CVPR2021](https://arxiv.org/abs/2104.00683v1)] SimPoE: Simulated Character Control for 3D Human Pose Estimation [[Project](https://www.ye-yuan.com/simpoe/)]
- [[Arxiv](https://arxiv.org/pdf/2104.00351v1.pdf)] TRAJEVAE - Controllable Human Motion Generation from Trajectories [[Project](https://kacperkan.github.io/trajevae-supplementary/)]
- [[CVPR2021](https://arxiv.org/abs/2103.17265v1)] Human POSEitioning System (HPS): 3D Human Pose Estimation and Self-localization in Large Scenes from Body-Mounted Sensors [[Project](http://virtualhumans.mpi-inf.mpg.de/hps/)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16449v1.pdf)] Bilevel Online Adaptation for Out-of-Domain Human Mesh Reconstruction [[Project](https://sites.google.com/view/humanmeshboa)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16341v1.pdf)] Learning Parallel Dense Correspondence from Spatio-Temporal Descriptors for Efficient and Robust 4D Reconstruction [[github](https://github.com/tangjiapeng/LPDC-Net)]
- [[Arxiv](https://arxiv.org/abs/2103.10978)] Probabilistic 3D Human Shape and Pose Estimation from Multiple Unconstrained Images in the Wild
- [[Arxiv](https://arxiv.org/pdf/2103.10455.pdf)] 3D Human Pose Estimation with Spatial and Temporal Transformers [[pytorch](https://github.com/zczcwh/PoseFormer)]
- [[CVPR2021](https://arxiv.org/abs/2103.10429)] Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks
- [[Arxiv](https://arxiv.org/pdf/2103.10206.pdf)] DanceNet3D: Music Based Dance Generation with Parametric Motion Transformer
- [[Arxiv](https://arxiv.org/pdf/2103.09755.pdf)] Aggregated Multi-GANs for Controlled 3D Human Motion Prediction [[Project](https://github.com/herolvkd/AM-GAN)]
- [[AAAI](https://arxiv.org/pdf/2103.09009.pdf)] PC-HMR: Pose Calibration for 3D Human Mesh Recovery from 2D Images/Videos
- [[Arxiv](https://arxiv.org/abs/2103.07700)] NeuralHumanFVV: Real-Time Neural Volumetric Human Performance Rendering using RGB Cameras
- [[CVPR2021](https://arxiv.org/pdf/2103.06871v1.pdf)] SMPLicit: Topology-aware Generative Model for Clothed People
 [[Project](http://www.iri.upc.edu/people/ecorona/smplicit/)]
- [[CVPR2021](https://arxiv.org/abs/2011.14672v2)] HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation [[pytorch](https://github.com/Jeff-sjtu/HybrIK)]
- [[Arxiv](https://arxiv.org/pdf/2103.00776.pdf)] Single-Shot Motion Completion with Transformer [[Project](https://github.com/FuxiCV/SSMCT)]
- [[EG2021](https://arxiv.org/abs/2103.00262)] Walk2Map: Extracting Floor Plans from Indoor Walk Trajectories
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
- [[ICCV2021](https://arxiv.org/abs/2008.12272)] Monocular, One-stage, Regression of Multiple 3D People [[github](https://github.com/Arthur151/ROMP)]
- [[ECCV2020](https://arxiv.org/pdf/2007.11755.pdf)] History Repeats Itself: Human Motion Prediction via Motion Attention [[pytorch](https://github.com/wei-mao-2019/HisRepItself)]
- [[ECCV2020](https://arxiv.org/abs/2007.13666)] 3D Human Shape and Pose from a Single Low-Resolution Image with Self-Supervised Learning [[Project](https://sites.google.com/view/xiangyuxu/3d_eccv20)]
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
- [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Putting_Humans_in_a_Scene_Learning_Affordance_in_3D_Indoor_CVPR_2019_paper.pdf)] Putting Humans in a Scene: Learning Affordance in 3D Indoor Environments [[Project](https://sites.google.com/view/3d-affordance-cvpr19)]
- [[TOG2016](https://graphics.stanford.edu/projects/pigraphs/pigraphs.pdf)] Pigraphs: learning interaction
snapshots from observations [[Project](https://graphics.stanford.edu/projects/pigraphs/)]














---
## General Methods
- [[Arxiv](https://arxiv.org/abs/2207.14455)] Neural Density-Distance Fields [[Project](https://ueda0319.github.io/neddf/)]
- [[Arxiv](https://arxiv.org/abs/2208.04164)] Understanding Masked Image Modeling via Learning Occlusion Invariant Feature
- [[Arxiv](https://arxiv.org/abs/2207.11971v1)] Jigsaw-ViT: Learning Jigsaw Puzzles in Vision Transformer [[Project](https://yingyichen-cyy.github.io/Jigsaw-ViT/)]
- [[Arxiv](https://arxiv.org/abs/2207.03111)] Masked Surfel Prediction for Self-Supervised Point Cloud Learning [[github](https://github.com/YBZh/MaskSurf)]
- [[Arxiv](https://arxiv.org/abs/2205.14401)] Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training [[github](https://github.com/ZrrSkywalker/Point-M2AE)]
- [[Arxiv](https://arxiv.org/abs/2206.14797)] 3D-Aware Video Generation [[Project](https://sherwinbahmani.github.io/3dvidgen/)]
- [[Arxiv](https://arxiv.org/abs/2206.11895)] Learning Viewpoint-Agnostic Visual Representations by Recovering Tokens in 3D Space [[Project](https://www3.cs.stonybrook.edu/~jishang/3dtrl/3dtrl.html)]
- [[Arxiv](https://arxiv.org/abs/2206.07706)] Masked Frequency Modeling for Self-Supervised Visual Pre-Training [[Project](https://www.mmlab-ntu.com/project/mfm/index.html)]
- [[Arxiv](https://arxiv.org/abs/2206.07255)] GRAM-HD: 3D-Consistent Image Generation at High Resolution with Generative Radiance Manifolds [[Project](https://jeffreyxiang.github.io/GRAM-HD/)]
- [[Arxiv](https://arxiv.org/abs/2206.07696)] Diffusion Models for Video Prediction and Infilling [[Project](https://arxiv.org/pdf/2206.07696.pdf)]
- [[Arxiv](https://arxiv.org/abs/2206.11894)] MaskViT: Masked Visual Pre-Training for Video Prediction [[Project](https://maskedvit.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2202.07453v1)] Random Walks for Adversarial Meshes
- [[ICLR2022](https://arxiv.org/abs/2202.07123v1)] Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework [[github](https://github.com/ma-xu/pointMLP-pytorch)]
- [[CVPR2022](https://arxiv.org/abs/2203.15102)] Rethinking Semantic Segmentation: A Prototype View [[github](https://github.com/tfzhou/ProtoSeg)]
- [[Arxiv](https://arxiv.org/abs/2202.03670)] How to Understand Masked Autoencoders
- [[ICLR2022](https://arxiv.org/abs/2201.02767v1)] QuadTree Attention for Vision Transformers [[github](https://github.com/Tangshitao/QuadtreeAttention)]
- [[Arxiv](https://arxiv.org/abs/2201.01922v1)] Contrastive Neighborhood Alignment
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.09343v1)] Domain Adaptation on Point Clouds via Geometry-Aware Implicits
- [[ICCV2021](https://arxiv.org/abs/2112.05213v1)] Progressive Seed Generation Auto-encoder for Unsupervised Point Cloud Learning
- [[Arxiv](https://arxiv.org/abs/2112.03777v1)] Variance-Aware Weight Initialization for Point Convolutional Neural Networks
- [[Arxiv](https://arxiv.org/abs/2112.01698v1)] Learning to Detect Every Thing in an Open World [[Project](https://ksaito-ut.github.io/openworld_ldet/)]
- [[Arxiv](https://arxiv.org/abs/2111.14819v1)] Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling [[Project](https://point-bert.ivg-research.xyz/)]
- [[Arxiv](https://arxiv.org/abs/2111.10866v1)] CpT: Convolutional Point Transformer for 3D Point Cloud Processing
- [[Arxiv](https://arxiv.org/abs/2111.09883v1)] Swin Transformer V2: Scaling Up Capacity and Resolution [[github](https://github.com/microsoft/Swin-Transformer)]
- [[Arxiv](https://arxiv.org/abs/2111.09833v1)] TransMix: Attend to Mix for Vision Transformers [[github](https://github.com/Beckschen/TransMix)]
- [[Arxiv](https://arxiv.org/abs/2111.06575v1)] Self-supervised GAN Detector [[github](https://github.com/ytongbai/ViTs-vs-CNNs)]
- [[NeurIPS2021](https://arxiv.org/abs/2110.15348v1)] Residual Relaxation for Multi-view Representation Learning
- [[ICCV2021](https://arxiv.org/abs/2110.02951v1)] Video Autoencoder: self-supervised disentanglement of static 3D structure and motion [[Project](https://zlai0.github.io/VideoAutoencoder/)]
- [[NeurIPS2021](https://arxiv.org/abs/2104.09125)] SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization [[Project](https://amirhertz.github.io/sape/)]
- [[Arxiv](https://matthew-a-chan.github.io/EG3D/media/eg3d.pdf)] Efficient Geometry-aware 3D Generative Adversarial Networks [[Project](https://matthew-a-chan.github.io/EG3D/)]
- [[Arxiv](https://arxiv.org/abs/2112.05682)] Self-attention Does Not Need $O(n^2)$ Memory
- [[Arxiv](https://arxiv.org/abs/2109.01291v1)] CAP-Net: Correspondence-Aware Point-view Fusion Network for 3D Shape Analysis
- [[Arxiv](https://arxiv.org/abs/2111.11187)] PointMixer: MLP-Mixer for Point Cloud Understanding
- [[NeurIPS2021](https://arxiv.org/abs/2110.15156)] Blending Anti-Aliasing into Vision Transformer
- [[ICCV2021](https://arxiv.org/abs/2108.12468v1)] Learning Inner-Group Relations on Point Clouds
- [[Arxiv](https://arxiv.org/abs/2108.06076v1)] Point-Voxel Transformer: An Efficient Approach To 3D Deep Learning
- [[Siggraph2021](https://arxiv.org/abs/2108.04476v1)] SP-GAN: Sphere-Guided 3D Shape Generation and Manipulation [[Project](https://liruihui.github.io/publication/SP-GAN/)] [[github](https://github.com/liruihui/SP-GAN)]
- [[ICCV2021](https://arxiv.org/abs/2108.00580v1)] GraphFPN: Graph Feature Pyramid Network for Object Detection
- [[Arxiv](https://arxiv.org/abs/2107.12655v1)] CKConv: Learning Feature Voxelization for Point Cloud Analysis
- [[ICCV2021](https://arxiv.org/abs/2103.15679)] Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers [[pytorch](https://github.com/hila-chefer/Transformer-MM-Explainability)]
- [[Arxiv](https://arxiv.org/pdf/2106.12052.pdf)] Volume Rendering of Neural Implicit Surfaces
- [[CVPR2021](https://igl.ethz.ch/projects/iso-points/iso_points-CVPR2021-yifan.pdf)] Iso-Points: Optimizing Neural Implicit Surfaces with Hybrid Representations
- [[Arxiv](https://arxiv.org/abs/2106.11795v1)] DeepMesh: Differentiable Iso-Surface Extraction
- [[Arxiv](https://arxiv.org/pdf/2106.11272v1.pdf)] Neural Marching Cubes
- [[Arxiv](https://arxiv.org/abs/2106.05187)] Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields
- [[Arxiv](https://arxiv.org/pdf/2106.02634.pdf)] Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering
- [[ICML2021](https://arxiv.org/pdf/2106.05304v1.pdf)] Revisiting Point Cloud Shape Classification with a Simple and Effective Baseline [[pytorch](https://github.com/princeton-vl/SimpleView)]
- [[Arxiv](https://arxiv.org/abs/2106.03804v1)] Deep Medial Fields
- [[Arxiv](https://arxiv.org/pdf/2106.02285v1.pdf)] Subdivision-Based Mesh Convolution Networks [[Jittor](https://github.com/lzhengning/SubdivNet)]
- [[Arxiv](https://arxiv.org/pdf/2106.00227.pdf)] VA-GCN: A Vector Attention Graph Convolution Network for learning on Point Clouds [[pytorch](https://github.com/hht1996ok/VA-GCN)]
- [[Arxiv](https://arxiv.org/pdf/2105.12723.pdf)] Aggregating Nested Transformers
- [[Arxiv](https://arxiv.org/abs/2105.07926)] Rethinking the Design Principles of Robust Vision Transformer [[pytorch](https://github.com/vtddggg/Robust-Vision-Transformer)]
- [[Siggraph2021](https://arxiv.org/pdf/2105.02788.pdf)] Acorn: Adaptive Coordinate Networks for Neural Scene Representation
- [[Arxiv](https://arxiv.org/pdf/2105.01288.pdf)] Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis [[Project](https://curvenet.github.io/)]
- [[Arxiv](https://arxiv.org/pdf/2105.08050.pdf)] Pay Attention to MLPs
- [[Arxiv](https://arxiv.org/pdf/2105.03404.pdf)] ResMLP: Feedforward networks for image classification with data-efficient training
- [[Arxiv](https://arxiv.org/abs/2105.01883)] RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
- [[Arxiv](https://arxiv.org/abs/2105.01601)] MLP-Mixer: An all-MLP Architecture for Vision
- [[Arxiv](https://arxiv.org/pdf/2104.12229.pdf)] Vector Neurons: A General Framework for SO(3)-Equivariant Networks
- [[CVPR2021](https://arxiv.org/pdf/2104.14554.pdf)] MongeNet: Efficient Sampler for Geometric Deep Learning [[Project](https://lebrat.github.io/MongeNet/)]
- [[Arxiv](https://arxiv.org/pdf/2104.13636.pdf)] Point Cloud Learning with Transformer
- [[Arxiv](https://arxiv.org/abs/2104.13044)] Dual Transformer for Point Cloud Analysis
- [[Arxiv](https://arxiv.org/pdf/2104.11571v1.pdf)] AttWalk: Attentive Cross-Walks for Deep Mesh Analysis
- [[Arxiv](https://arxiv.org/pdf/2104.04687.pdf)] Learning from 2D: Pixel-to-Point Knowledge Transfer for 3D Pretraining
- [[Arxiv](https://arxiv.org/pdf/2104.03916v1.pdf)] Field Convolutions for Surface CNNs
- [[Arxiv](https://arxiv.org/pdf/2103.16302v1.pdf)] Rethinking Spatial Dimensions of Vision Transformers [[pytorch](https://github.com/naver-ai/pit)] :fire:
- [[CVPR2021](https://arxiv.org/pdf/2103.14635v1.pdf)] PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds [[pytorch](https://github.com/CVMI-Lab/PAConv)]
- [[Arxiv](https://arxiv.org/pdf/2103.10484.pdf)] Concentric Spherical GNN for 3D Representation Learning
- [[Arxiv](https://arxiv.org/abs/2102.06171)] High-Performance Large-Scale Image Recognition Without Normalization
- [[Arxiv](https://arxiv.org/pdf/2102.04776.pdf)] Generative Models as Distributions of Functions
- [[Arxiv](https://arxiv.org/pdf/2102.04014.pdf)] Point-set Distances for Learning Representations of 3D Point Clouds
- [[Arxiv](https://arxiv.org/pdf/2102.02896.pdf)] Compressed Object Detection
- [[Arxiv](https://arxiv.org/pdf/2102.00084.pdf)] A linearized framework and a new benchmark for model selection for fine-tuning
- [[Arxiv](https://arxiv.org/pdf/2101.07832.pdf)] The Devils in the Point Clouds: Studying the Robustness of Point Cloud Convolutions
- [[Arxiv](https://arxiv.org/pdf/2101.02691.pdf)] Self-Supervised Pretraining of 3D Features on any Point-Cloud [[pytorch](https://github.com/facebookresearch/DepthContrast)]
- [[3DV2020](https://arxiv.org/pdf/2101.00483.pdf)] Learning Rotation-Invariant Representations of Point Clouds Using Aligned Edge Convolutional Neural Networks

#### Before 2021
- [[ICCV2019](https://arxiv.org/abs/1908.09186)] Efficient Learning on Point Clouds with Basis Point Sets [[pytorch](https://github.com/sergeyprokudin/bps)]
- [[CVPR2019](https://arxiv.org/abs/1812.07035)] On the Continuity of Rotation Representations in Neural Networks [[pytorch](https://github.com/papagina/RotationContinuity)]
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
- [[CVPR2022](https://arxiv.org/abs/2204.05145)] Focal Length and Object Pose Estimation via Render and Compare [[github](https://github.com/ponimatkin/focalpose)]
- [[CVPR2022](https://arxiv.org/abs/2203.03570)] Kubric: A scalable dataset generator
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.13047v1)] Channel-Wise Attention-Based Network for Self-Supervised Monocular Depth Estimation
- [[Arxiv](https://arxiv.org/abs/2112.02306v1)] Toward Practical Self-Supervised Monocular Indoor Depth Estimation
- [[Arxiv](https://arxiv.org/abs/2112.00933v1)] PartImageNet: A Large, High-Quality Dataset of Parts [[github](https://github.com/TACJu/PartImageNet)]
- [[Arxiv](https://arxiv.org/abs/2112.00246v1)] AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-shot Interactions
- [[Arxiv](https://arxiv.org/abs/2111.11429v1)] Benchmarking Detection Transfer Learning with Vision Transformers
- [[Arxiv](https://arxiv.org/abs/2111.10250v1)] Panoptic Segmentation: A Review [[github](https://github.com/elharroussomar/Awesome-Panoptic-Segmentation)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.07383v1)] Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space [[github](https://github.com/Gorilla-Lab-SCUT/SS-Conv)]
- [[Arxiv](https://arxiv.org/abs/2111.07624v1)] Attention Mechanisms in Computer Vision: A Survey
- [[Arxiv](https://arxiv.org/abs/2111.05615v1)] Leveraging Geometry for Shape Estimation from a Single RGB Image [[github](https://github.com/florianlanger/leveraging_geometry_for_shape_estimation)]
- [[Arxiv](https://arxiv.org/abs/2111.02045v1)] Deep Point Set Resampling via Gradient Fields [[github](https://github.com/luost26/score-denoise)]
- [[Arxiv](https://arxiv.org/abs/2111.02135v1)] Efficient 3D Deep LiDAR Odometry [[github](https://github.com/IRMVLab/PWCLONet)]
- [[NeurIPS2021](https://arxiv.org/abs/2111.00312v1)] 3DP3: 3D Scene Perception via Probabilistic Programming
- [[NeurIPS2021](https://arxiv.org/abs/2110.14076v1)] CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration [[github](https://github.com/haoyu94/Coarse-to-fine-correspondences)]
- [[BMVC2021](https://arxiv.org/abs/2110.12204v1)] Cascading Feature Extraction for Fast Point Cloud Registration
- [[Arxiv](https://arxiv.org/abs/2110.11545v1)] Pseudo Supervised Monocular Depth Estimation with Teacher-Student Network
- [[BMVC2021](https://arxiv.org/abs/2110.11608v1)] Multi-Stream Attention Learning for Monocular Vehicle Velocity and Inter-Vehicle Distance Estimation
- [[Arxiv](https://arxiv.org/abs/2110.11636v1)] Occlusion-Robust Object Pose Estimation with Holistic Representation [[github](https://github.com/BoChenYS/ROPE)]
- [[BMVC2021](https://arxiv.org/abs/2110.11679v1)] Depth-only Object Tracking
- [[3DV2021](https://arxiv.org/abs/2110.11275v1)] Self-Supervised Monocular Scene Decomposition and Depth Estimation
- [[Arxiv](https://arxiv.org/abs/2110.10494v1)] Deep Point Cloud Normal Estimation via Triplet Learning
- [[3DV2021](https://arxiv.org/abs/2110.08192v1)] Attention meets Geometry: Geometry Guided Spatial-Temporal Attention for Consistent Self-Supervised Monocular Depth Estimation
- [[CORL2021](https://arxiv.org/abs/2110.06558v1)] LENS: Localization enhanced by NeRF synthesis
- [[3DV2021](https://arxiv.org/abs/2110.05839v1)] PLNet: Plane and Line Priors for Unsupervised Indoor Depth Estimation [[github](https://github.com/HalleyJiang/PLNet)]
- [[Arxiv](https://arxiv.org/abs/2110.04411v1)] Unsupervised Pose-Aware Part Decomposition for 3D Articulated Objects
- [[ICCV2021](https://arxiv.org/abs/2110.01269v1)] PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds [[Project](https://github.com/valeoai/PCAM)]
- [[ICCV2021](https://arxiv.org/abs/2109.12484v1)] Excavating the Potential Capacity of Self-Supervised Monocular Depth Estimation
- [[ICCV2021](https://arxiv.org/abs/2109.10115v2)] StereOBJ-1M: Large-scale Stereo Image Dataset for 6D Object Pose Estimation
- [[IROS2021](https://arxiv.org/abs/2109.10127v1)] KDFNet: Learning Keypoint Distance Field for 6D Object Pose Estimation
- [[ICCV2021](https://arxiv.org/abs/2109.09881v1)] Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation [[github](https://github.com/baegwangbin/surface_normal_uncertainty)]
- [[Arxiv](https://arxiv.org/abs/2111.00190)] Leveraging SE(3) Equivariance for Self-Supervised Category-Level Object Pose Estimation [[Project](https://dragonlong.github.io/equi-pose/)]
- [[ICCV2021](https://arxiv.org/abs/2109.04310v1)] Deep Hough Voting for Robust Global Registration
- [[Arxiv](https://arxiv.org/abs/2109.00182v1)] You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors [[Project](https://hpwang-whu.github.io/YOHO/)]
- [[ICCV2021](https://arxiv.org/abs/2108.11682v1)] A Robust Loss for Point Cloud Registration
- [[Arxiv](https://arxiv.org/abs/2108.09169v1)] Geometry-Aware Self-Training for Unsupervised Domain Adaptationon Object Point Clouds
- [[IROS2021](https://arxiv.org/abs/2108.08755v1)] Category-Level 6D Object Pose Estimation via Cascaded Relation and Recurrent Reconstruction Networks [[Project](https://wangjiaze.cn/projects/6DPoseEstimation.html)] [[github](https://github.com/JeremyWANGJZ/Category-6D-Pose)]
- [[ICCV2021](https://arxiv.org/abs/2108.08574v1)] StructDepth: Leveraging the structural regularities for self-supervised indoor depth estimation [[github](https://github.com/SJTU-ViSYS/StructDepth)]
- [[ICCV2021](https://arxiv.org/abs/2108.08367v1)] SO-Pose: Exploiting Self-Occlusion for Direct 6D Pose Estimation
- [[ICCV2021](https://arxiv.org/abs/2108.07628v1)] Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation
- [[ICCV2021](https://arxiv.org/abs/2108.05836v1)] AdaFit: Rethinking Learning-based Normal Estimation on Point Clouds [[Project](https://runsong123.github.io/AdaFit/)]
- [[Arxiv](https://arxiv.org/abs/2108.05615v1)] DnD: Dense Depth Estimation in Crowded Dynamic Indoor Scenes
- [[ICCV2021](https://arxiv.org/abs/2108.05312v1)] Towards Interpretable Deep Networks for Monocular Depth Estimation [[github](https://github.com/youzunzhi/InterpretableMDE)]
- [[Arxiv](https://arxiv.org/abs/2108.02740v1)] UPDesc: Unsupervised Point Descriptor Learning for Robust Registration
- [[IROS2021](https://arxiv.org/abs/2108.00516v1)] BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models [[github](https://github.com/wenbowen123/BundleTrack)]
- [[Arxiv](https://arxiv.org/abs/2107.13802v1)] RigNet: Repetitive Image Guided Network for Depth Completion
- [[Arxiv](https://arxiv.org/abs/2107.13087v1)] DCL: Differential Contrastive Learning for Geometry-Aware Depth Synthesis
- [[ACMMM2021](https://arxiv.org/abs/2107.12541v1)] BridgeNet: A Joint Learning Network of Depth Map Super-Resolution and Monocular Depth Estimation [[Project](https://rmcong.github.io/proj_BridgeNet.html)] [[github](https://github.com/rmcong/BridgeNet_ACM-MM-2021)]
- [[Arxiv](https://arxiv.org/abs/2107.12549v1)] Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation
- [[ICCV2021](https://arxiv.org/abs/2107.11992)] HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration [[Project](https://ispc-group.github.io/hregnet)] [[pytorch](https://github.com/ispc-lab/HRegNet)]
- [[Arxiv](https://arxiv.org/abs/2107.10981v1)] Score-Based Point Cloud Denoising
- [[Arxiv](https://arxiv.org/abs/2107.03180v1)] HIDA: Towards Holistic Indoor Understanding for the Visually Impaired via Semantic Instance Segmentation with a Wearable Solid-State LiDAR Sensor
- [[Arxiv](https://arxiv.org/abs/2107.02972v1)] Learn to Learn Metric Space for Few-Shot Segmentation of 3D Shapes
- [[Arxiv](https://arxiv.org/pdf/2106.08615v1.pdf)] EdgeConv with Attention Module for Monocular Depth Estimation
- [[ICML2021](https://arxiv.org/pdf/2106.05965v1.pdf)] Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold [[Project](https://implicit-pdf.github.io/)]
- [[ICRA2021](https://arxiv.org/pdf/2106.03010v1.pdf)] An Adaptive Framework For Learning Unsupervised Depth Completion [[github](https://github.com/alexklwong/adaframe-depth-completion)] [[github](https://github.com/alexklwong/learning-topology-synthetic-data)]
- [[ICRA2021](https://arxiv.org/abs/2105.07468)] TSDF++: A Multi-Object Formulation for Dynamic Object Tracking and Reconstruction [[github](https://github.com/ethz-asl/tsdf-plusplus)]
- [[Siggraph2021](https://arxiv.org/abs/2105.01604)] Orienting Point Clouds with Dipole Propagation
- [[CVPR2021](https://arxiv.org/abs/2104.14540)] The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth
- [[Arxiv](https://arxiv.org/pdf/2104.11207.pdf)] Fully Convolutional Line Parsing [[pytorch](https://github.com/Delay-Xili/F-Clip)]
- [[CVPR2021](https://arxiv.org/pdf/2104.07350.pdf)] Depth Completion using Plane-Residual Representation
- [[Arxiv](https://arxiv.org/pdf/2104.05764.pdf)] Domain Adaptive Monocular Depth Estimation With Semantic Information
- [[CVPR2021](https://arxiv.org/pdf/2104.02253v2.pdf)] Depth Completion with Twin Surface Extrapolation at Occlusion Boundaries [[github](https://github.com/imransai/TWISE)]
- [[Arxiv](https://arxiv.org/pdf/2104.02631v1.pdf)] Local Metrics for Multi-Object Tracking
- [[Arxiv](https://arxiv.org/pdf/2104.00152v1.pdf)] Full Surround Monodepth from Multiple Cameras
- [[CVPR2021](https://arxiv.org/abs/2104.00622v1)] RGB-D Local Implicit Function for Depth Completion of Transparent Objects [[Project](https://research.nvidia.com/publication/2021-03_RGB-D-Local-Implicit)]
- [[CVPR2021](https://arxiv.org/pdf/2103.16792v1.pdf)] Learning Camera Localization via Dense Scene Matching [[pytorch](https://github.com/Tangshitao/Dense-Scene-Matching)]
- [[Arxiv](https://arxiv.org/pdf/2103.15039v1.pdf)] LSG-CPD: Coherent Point Drift with Local Surface Geometry for Point Cloud Registration
- [[ICRA2021](https://arxiv.org/abs/2103.15428v1)] PlaneSegNet: Fast and Robust Plane Estimation Using a Single-stage Instance Segmentation CNN
- [[Arxiv](https://arxiv.org/pdf/2103.13030v1.pdf)] Learning Fine-Grained Segmentation of 3D Shapes without Part Labels
- [[CVPR2021](https://arxiv.org/pdf/2103.10814.pdf)] Skeleton Merger: an Unsupervised Aligned Keypoint Detector
- [[CVPR2021](https://arxiv.org/pdf/2103.08468.pdf)] Beyond Image to Depth: Improving Depth Prediction using Echoes
- [[CVPR2021](https://arxiv.org/abs/2103.07054)] FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism [[Project](https://github.com/DC1991/FS-Net)]
- [[CVPR2021](https://arxiv.org/abs/2103.03114)] Self-supervised Geometric Perception
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
- [[NeurIPS2019](https://arxiv.org/pdf/1910.12240.pdf)] PRNet: Self-Supervised Learning for Partial-to-Partial Registration [[pytorch](https://github.com/WangYueFt/prnet)]
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
- [[NeurIPS2021](https://openreview.net/pdf?id=tjZjv_qh_CE)] ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data [[github](https://github.com/apple/ARKitScenes)]
- [[Dataset](https://aihabitat.org/datasets/replica_cad/)] ReplicaCAD [[Project](https://aihabitat.org/datasets/replica_cad/)]
- [[PhDthesis](https://arxiv.org/abs/2202.12752v1)] Synthesizing Photorealistic Images with Deep Generative Learning
- [[ICCVW2021](https://arxiv.org/abs/2202.08449v1)] V2X-Sim: A Virtual Collaborative Perception Dataset for Autonomous Driving [[Project](https://ai4ce.github.io/V2X-Sim/)]
- [[Arxiv](https://arxiv.org/abs/2202.08471v1)] TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and Grasping [[Project](https://graspnet.net/transcg)]
- [[Arxiv](https://arxiv.org/abs/2202.07183v1)] A Survey of Neural Trojan Attacks and Defenses in Deep Learning
- [[Arxiv](https://arxiv.org/abs/2202.05659v1)] Tiny Object Tracking: A Large-scale Dataset and A Baseline [[github](https://github.com/mmic-lcl/Datasets-and-benchmark-code)]
- [[Arxiv](https://arxiv.org/abs/2202.02656v1)] A survey of top-down approaches for human pose estimation
- [[Arxiv](https://arxiv.org/abs/2201.05761v1)] A Survey on RGB-D Datasets
- [[Arxiv](https://arxiv.org/abs/2201.03299v1)] Avoiding Overfitting: A Survey on Regularization Methods for Convolutional Neural Networks
#### Before 2022
- [[Arxiv](https://arxiv.org/abs/2112.12988v1)] iSeg3D: An Interactive 3D Shape Segmentation Tool
- [[Arxiv](https://arxiv.org/abs/2112.13018v1)] Benchmarking Pedestrian Odometry: The Brown Pedestrian Odometry Dataset (BPOD) [[Project](https://repository.library.brown.edu/studio/item/bdr:p52vqgtg/)]
- [[Arxiv](https://arxiv.org/abs/2112.12610v1)] PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving [[Project](https://scale.com/open-datasets/pandaset)]
- [[Arxiv](https://arxiv.org/abs/2112.11699v1)] Few-Shot Object Detection: A Survey
- [[Arxiv](https://arxiv.org/abs/2111.11348v1)] Paris-CARLA-3D: A Real and Synthetic Outdoor Point Cloud Dataset for Challenging Tasks in 3D Mapping [[Project](https://npm3d.fr/paris-carla-3d)]
- [[Arxiv](https://arxiv.org/abs/2111.09887v1)] PyTorchVideo: A Deep Learning Library for Video Understanding [[Project](https://pytorchvideo.org/)]
- [[Arxiv](https://arxiv.org/abs/2110.11590v1)] DIML/CVL RGB-D Dataset: 2M RGB-D Images of Natural Indoor and Outdoor Scenes [[Project](https://dimlrgbd.github.io/)]
- [[Arxiv](https://arxiv.org/abs/2110.06877v1)] A Review on Human Pose Estimation
- [[ICCV2021](https://arxiv.org/abs/2110.04955v1)] BuildingNet: Learning to Label 3D Buildings [[Project](https://buildingnet.org/)]
- [[ICCV2021](https://arxiv.org/abs/2110.04994v1)] Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans [[Project](https://omnidata.vision/)]
- [[Arxiv](https://arxiv.org/abs/2109.08238v1)] Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI
- [[Arxiv](https://arxiv.org/abs/2107.06149)] MINERVAS: Massive INterior EnviRonments VirtuAl Synthesis [[Project](https://coohom.github.io/MINERVAS/)]
- [[Arxiv](https://arxiv.org/abs/2107.04286v1)] UrbanScene3D: A Large Scale Urban Scene Dataset and Simulator [[Project](https://vcc.tech/UrbanScene3D/)]
- [[Arxiv](https://arxiv.org/abs/2106.11118v2)] SODA10M: Towards Large-Scale Object Detection Benchmark for Autonomous Driving [[Project](https://soda-2d.github.io/documentation.html#data_collection)]
- [[Arxiv](https://arxiv.org/pdf/2106.11650v1.pdf)] A Survey on Human-aware Robot Navigation
- [[Arxiv](https://arxiv.org/pdf/2106.11037v1.pdf)] One Million Scenes for Autonomous Driving: ONCE Dataset [[Project](https://once-for-auto-driving.github.io/index.html)]
- [[Arxiv](https://arxiv.org/abs/2106.10823v1)] 3D Object Detection for Autonomous Driving: A Survey
- [[Arxiv](https://arxiv.org/pdf/2106.08983v1.pdf)] The Oxford Road Boundaries Dataset
- [[CVPR2021](https://arxiv.org/abs/2103.16397)] 3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding
- [[Arxiv](https://arxiv.org/abs/2106.03805)] 3DB: A Framework for Debugging Computer Vision Models [[github](https://github.com/3db/3db)]
- [[Arxiv](https://arxiv.org/abs/2105.13962)] NViSII: A Scriptable Tool for Photorealistic Image Generation [[github](https://github.com/owl-project/NVISII)]
- [[Dataset](https://github.com/bertjiazheng/Structured3D)] Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling
- [[Survey](https://arxiv.org/abs/2103.07466v2)] 3D Semantic Scene Completion: a Survey
- [[Survey](https://arxiv.org/pdf/2103.05423v2.pdf)] Deep Learning based 3D Segmentation: A Survey
- [[Survey](https://arxiv.org/pdf/2103.02690.pdf)] A comprehensive survey on point cloud registration
- [[Survey](https://arxiv.org/pdf/2103.02503.pdf)] Domain Generalization: A Survey
- [[Dataset](https://arxiv.org/pdf/2103.00355.pdf)] SUM: A Benchmark Dataset of Semantic Urban Meshes
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
- [[Dataset](https://arxiv.org/abs/1906.05797)] The Replica Dataset: A Digital Replica of Indoor Spaces [[github](https://github.com/facebookresearch/Replica-Dataset)]
- [[IROS2021](https://arxiv.org/abs/2012.02924)] iGibson 1.0: a Simulation Environment for Interactive Tasks in Large Realistic Scenes [[Project](http://svl.stanford.edu/igibson/)]
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
- [[Github](https://github.com/vchoutas/torch-mesh-isect)] Detecting & Penalizing Mesh Intersections
- [[CVPR2021](https://arxiv.org/pdf/2103.15076v1.pdf)] Picasso: A CUDA-based Library for Deep Learning over 3D Meshes
 [[Github](https://github.com/hlei-ziyan/Picasso)]
- [[Github](https://github.com/NVIDIA/DALI)] A GPU-accelerated library containing highly optimized building blocks and an execution engine for data processing to accelerate deep learning training and inference applications
- [[Arxiv](https://arxiv.org/pdf/2104.05125.pdf)] Shuffler: A Large Scale Data Management Tool for Machine Learning in Computer Vision
- [[Arxiv](https://arxiv.org/abs/2106.06158v1)] PyGAD: An Intuitive Genetic Algorithm Python Library [[Github](https://github.com/ahmedfgad/GeneticAlgorithmPython)]
- [[Arxiv](https://arxiv.org/abs/2106.06158v1)] PyGAD: An Intuitive Genetic Algorithm Python Library [[Github](https://github.com/ahmedfgad/GeneticAlgorithmPython)]
- [[ICRA2014](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf)] A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and
SLAM [[Project](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)]
- [[CVPR2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Handa_Understanding_Real_World_CVPR_2016_paper.pdf)] SceneNet: Understanding Real World Indoor Scenes With Synthetic Data [[Project](https://robotvault.bitbucket.io/)]

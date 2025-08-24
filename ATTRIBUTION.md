# Attribution

## 📣 Model & Acknowledgements

This project uses the **YOLOv8 pothole/road-damage segmentation model** provided by Farzad Nekouee in the repository:

```bash
https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment
```

The pretrained model from that repository (`model/best.pt`) is used here to detect potholes and damage masks, which feed into the heatmap and path-planning logic.  
A big thanks to the author for releasing this model — it made the pothole detection pipeline much easier to integrate.



## 🔗 Third-Party Components

### YOLOv8 Pothole Segmentation & Road Damage Assessment
- **Repository:** https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment  
- **Usage:** Provides pretrained weights and segmentation logic for pothole detection.  
- **License:** See upstream LICENSE file in the linked repository.  

### Ultralytics YOLOv8
- **Repository:** https://github.com/ultralytics/ultralytics  
- **Usage:** Core model runtime for segmentation and inference.  
- **License:** See upstream LICENSE file in the linked repository.  

### Other Libraries
- **OpenCV** (`opencv-python`) — image processing & visualization  
- **NumPy** — numerical operations  
- **PyTorch** — model runtime backend  
- **SciPy** — interpolation utilities  


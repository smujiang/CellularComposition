/**
 * * Create a region annotation by enlarging the original area a little bit to fit the patch extraction size
 *
 * @author Jun Jiang  Jiang.Jun@mayo.edu
 */
// TODO: need to be fixed. If the ROI is locked, this code can't change the size of it.
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
//import qupath.lib.scripting.QPEx

// the patch size you would like to extract from ROIs
def patch_size = [512, 512]

// Get the project currently opened in QuPath
def project = QPEx.getProject()
if (project == null) {
    print 'No project open!'
    return
}

// enlarge the detected area a little bit to fit the patch extraction size
def adjustROIforProperExtraction(List rect, List patch_size, List img_size){
    w_m = rect[2] % patch_size[0]
    h_m = rect[3] % patch_size[1]
    def new_w = 0
    def new_h = 0
    for (i in 0..patch_size[0]){
        new_w = rect[2]+i
        if ((new_w) % patch_size[0] == 0){
            break
        }
    }
    for (i in 0..patch_size[1]){
        new_h = rect[3]+i
        if ((new_h) % patch_size[1] == 0){
            break
        }
    }
    return [rect[0], rect[1], new_w, new_h]
}

//
for (entry in project.getImageList()) {
    // Get main data structures
    print(sprintf("Processing %s", entry.getImageName()))
//    def imageData = QPEx.getCurrentImageData()
    def imageData = entry.readImageData()
    def server = imageData.getServer()
    def Anno_list = []
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
    int idx = 0
    for (anno in annotations){
        if(anno.getPathClass().toString().equals("ROI")){
            Anno_list << anno
            // remove the old annotation
//            imageData.getHierarchy().removeObjects(anno, false)
            imageData.getHierarchy().removeObject(annotations.getAt(idx), false)
        }
        idx += 1
    }
    if (server.getPixelCalibration().hasPixelSizeMicrons()){
//    if (server.hasPixelSizeMicrons()){
        for (anno in Anno_list){
            // TODO: optimize the size of ROI
            double cx = (int)anno.getROI().x
            double cy = (int)anno.getROI().y
            double w = (int)(anno.getROI().x2 - cx)
            double h = (int)(anno.getROI().y2 - cy)
            def xy = [cx, cy, w, h]
            
            def WSI_Width = server.width
            def WSI_Height = server.height
            def rect = adjustROIforProperExtraction(xy, patch_size, [WSI_Width, WSI_Height])
            print(sprintf("\t Original size:[x=%d, y=%d, w=%d, h=%d]", (int)xy[0], (int)xy[1], (int)xy[2], (int)xy[3]))
            print(sprintf("\t Optimized size:[x=%d, y=%d, w=%d, h=%d]",(int)rect[0], (int)rect[1], (int)rect[2], (int)rect[3]))
            def roi = new RectangleROI(cx, cy, rect[2], rect[3])

            // Create & new annotation & add it to the object hierarchy
            def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("ROI"))
            annotation.setLocked(true) //Lock this annotation
            imageData.getHierarchy().addPathObject(annotation, false)
            QPEx.fireHierarchyUpdate()
        }
    }
    else{
        return
    }
}
print("Done")



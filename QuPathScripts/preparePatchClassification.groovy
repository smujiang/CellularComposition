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
////////////////////////////////////////////
// Process ROIs
// enlarge the detected area a little bit to fit the patch extraction size
////////////////////////////////////////////
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
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
def hierarchy = QPEx.getCurrentHierarchy()
// Get main data structures
print(sprintf("Processing %s", QPEx.name))
//    def imageData = QPEx.getCurrentImageData()

def Anno_list = []

def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
int idx = 0
for (anno in annotations){
    if(anno.getPathClass().toString().equals("ROI")){
        Anno_list << anno
        // remove the old annotation
//            imageData.getHierarchy().removeObjects(anno, false)
        annotations.getAt(idx).setLocked(false)
        imageData.getHierarchy().removeObject(annotations.getAt(idx), false)
    }
    idx += 1
}
if (server.getPixelCalibration().hasPixelSizeMicrons()){
//    if (server.hasPixelSizeMicrons()){
    for (anno in Anno_list){
        // TODO: optimize the size of ROI
        int cx = (int)anno.getROI().x
        int cy = (int)anno.getROI().y
        int w = (int)(anno.getROI().x2 - cx)
        int h = (int)(anno.getROI().y2 - cy)
        def xy = [cx, cy, w, h]

        def WSI_Width = server.width
        def WSI_Height = server.height
        def rect = adjustROIforProperExtraction(xy, patch_size, [WSI_Width, WSI_Height])
        print(sprintf("\t Original size:[x=%d, y=%d, w=%d, h=%d]", (int)xy[0], (int)xy[1], (int)xy[2], (int)xy[3]))
        print(sprintf("\t Optimized size:[x=%d, y=%d, w=%d, h=%d]",(int)rect[0], (int)rect[1], (int)rect[2], (int)rect[3]))
        def roi = new RectangleROI(cx, cy, rect[2], rect[3])

        print(sprintf("\t Can get %d patches from this ROI", (int)(rect[2]/patch_size[0]+ rect[3]/patch_size[1])))
        // Create & new annotation & add it to the object hierarchy
        def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("ROI"))
//        annotation.setLocked(true) //Lock this annotation
        imageData.getHierarchy().addPathObject(annotation, false)
//        QPEx.fireHierarchyUpdate()
    }
}
else{
    return
}

print("Process ROIs Done!")
////////////////////////////////////////////
// Run cell detection
////////////////////////////////////////////
//def ROI_list = []
//for (anno in annotations){
//    if(anno.getPathClass().toString().equals("ROI")){
//        ROI_list << anno
//    }
//}

selectAnnotations()
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection',
        '{"requestedPixelSizeMicrons": 0.25, "backgroundRadiusMicrons": 8, "medianRadiusMicrons": 0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "watershedPostProcess": true,  "cellExpansionMicrons": 1.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');
//def detections = getDetectionObjects()
//print(detections)
//for (d in detections){
//    if (! d.pathClass){
//        d.setPathClass(PathClassFactory.getPathClass("Tumor"))
//    }
//    QPEx.getCurrentImageData().getHierarchy().addPathObject(d, false)
//}
//
//QPEx.fireHierarchyUpdate()


////////////////////////////////////////////
// Save detection measurements
////////////////////////////////////////////
//output_dir = "/Users/m192500/Dataset/OvaryCancer/CellClassification"
output_dir = "H:\\OvaryCancer\\ImageData\\QuPathMeasurements_for_patchLevel"

pathOutput = output_dir + File.separator + server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
File newDir = new File(pathOutput)
if (!newDir.exists()) {
    newDir.mkdirs()
}
save_fn = pathOutput + File.separator + "detections_measurements.txt"
print(save_fn)
saveDetectionMeasurements(save_fn)


print("Done")





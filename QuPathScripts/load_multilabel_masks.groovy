/**
 * * load multi-label mask images (usually from prediction) into QuPath, so as to have better visualization and interaction
 *
 * @author Jun Jiang  Jiang.Jun@mayo.edu
 */
import ij.measure.Calibration
import ij.plugin.filter.ThresholdToSelection
import ij.process.ByteProcessor
import ij.process.ColorProcessor
import ij.process.FloatProcessor
import ij.process.ImageProcessor

import ij.blob.*
import qupath.imagej.tools.IJTools
import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.roi.ROIs

//import qupath.imagej.objects.ROIConverterIJ
//import qupath.imagej.tools.IJTools
import qupath.lib.roi.RoiTools

//import qupath.lib.scripting.QP
import qupath.lib.gui.scripting.QPEx
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import javax.imageio.ImageIO
import java.awt.Color
import java.awt.Image
import java.awt.image.BufferedImage
import java.awt.image.Raster
import ij.ImagePlus
//import inra.ijpb.label.LabelImages

// TODO: 1. get rid of zeros in allBlobs.get(conn).outerContour.xpoints and ypoints
// TODO: 2. load each connected component as a cell (PathCellObject in QuPath), not Annotation or Detection in QuPath
// TODO: 3. load class_txt_list, class_color_list from outside files;
//

// List<String> class_txt_list = ["Tumor", "Lymphocyte", "Stroma", "Non-Stroma"]
// List<List<Integer>> class_color_list = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 0, 0]]

List<String> class_txt_list = ["Tumor", "Lymphocyte", "Stroma", "Macrophage", "Karyorrhexis", "RBC"]
List<List<Integer>> class_color_list = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 148, 255], [255, 0, 255]]

def Load_As = "Annotation"  // could also be "Detection"
//def Load_As = "Detection"

def hierarchy = QPEx.getCurrentHierarchy()

//img_fn = "/Users/m192500/Dataset/OvaryData/QBRC_all/512/OCMC-001_mask/OCMC-001_35972_33137.png"
//print(img_fn)
//mask_dir = "H:\\OvaryCancer\\CellClassification\\ROI_Masks_out_norm_out\\OCMC-020"
mask_dir = "/Users/m192500/Dataset/OvaryCancer/CellClassification/ROI_Masks_out/OCMC-016"
//mask_dir = "/Users/m192500/Dataset/OvaryData/Patches_out"
def dirMask = new File(mask_dir)
if (!dirMask.isDirectory()) {
    print dirMask + ' is not a valid directory!'
    return
}
def files = dirMask.listFiles({f -> f.isFile() && f.getName().endsWith('.png') } as FileFilter) as List
if (files.isEmpty()) {
    print 'No mask files found in ' + mask_dir
    return
}
files.each {
    try {
        def all_annotations = parseMultilabelMask(it, class_color_list, class_txt_list, Load_As)
        hierarchy.addPathObjects(all_annotations)
    } catch (Exception e) {
        print 'Unable to parse annotation from ' + it.getName() + ': ' + e.getLocalizedMessage()
    }
}
//def all_annotations = parseMultilabelAnnotation(new File(img_fn), class_color_list, class_txt_list)

//def all_classes = QPEx.getQuPath().availablePathClasses
//for(anno in all_annotations){
//    hierarchy.addPathObject(anno)
//}
QPEx.fireHierarchyUpdate()
/**
 * Create a new annotation from a binary image, parsing the classification & region from the file name.
 *
 * Note: this code doesn't bother with error checking or handling potential issues with formatting/blank images.
 * If something is not quite right, it is quite likely to throw an exception.
 *
 * @param file File containing the PNG image mask.  The image name must be formatted as above.
 * @return The PathAnnotationObject created based on the mask & file name contents.
 */
static def parseMultilabelMask(File file, List<List<Integer>> class_color_list, List<String> class_txt_list, String Load_As) {
    def annotations = []
    def img = ImageIO.read(file)
    // get original coordinate from image file name
    def end = file.name.lastIndexOf("_")
    //def end_r = file.name.lastIndexOf("-mask.png")
    def end_r = file.name.lastIndexOf(".png")
    def start = file.name.indexOf("_")
    char[] buf = new char[end - start-1]
    file.name.getChars(start+1, end, buf,0)
    int orig_x = Integer.parseInt(buf.toString())
    buf = new char[end_r - end - 1]
    file.name.getChars(end+1, end_r, buf,0)
    int orig_y = Integer.parseInt(buf.toString())
    for(int col_idx = 0; col_idx < class_color_list.size(); col_idx++){
        def color = class_color_list[col_idx]
        def binary_img = getColorSegmentation(img, color[0], color[1], color[2], 0)
        def bi = binary_img.getBufferedImage()
        def imp = new ImagePlus(class_txt_list[col_idx], bi)
        ManyBlobs allBlobs = new ManyBlobs(imp)
        allBlobs.setBackground(0)  // 0 for black, 1 for 255
        allBlobs.findConnectedComponents() // Start the Connected Component Algorithm
        for(int conn = 0; conn < allBlobs.size(); conn++){
            def x = allBlobs.get(conn).outerContour.xpoints
            def y = allBlobs.get(conn).outerContour.ypoints
//            print("\n")
//            print("\n")
//            print(x)
//            print("\n")
//            print(y)
//            print("\n")
//            print("\n")
            def tmp_points_list = []
            for(int i = 0; i < x.size(); i++){
                if(x[i] != 0 && y[i] != 0){
                    tmp_points_list.add(new Point2(x[i]+orig_x, y[i]+orig_y))
                }
            }
            print(tmp_points_list)
            def roi = new PolygonROI(tmp_points_list)
            if(Load_As == "Annotation"){
                def annotation = new PathAnnotationObject(roi)
                annotation.setPathClass(QPEx.getPathClass(class_txt_list[col_idx]))
                annotations.add(annotation)
            }
            else if(Load_As == "Detection"){
                def annotation = new PathDetectionObject(roi)
                annotation.setPathClass(QPEx.getPathClass(class_txt_list[col_idx]))
                annotations.add(annotation)
            }
            else{
                throw new Exception("Undefined mask loading option ")
            }
        }
    }
    return annotations
}

static def getColorSegmentation(BufferedImage image, int r, int g, int b, int tolerance){
    // Pre-calc RGB "tolerance" values out of the loop (min is 0 and max is 255)
    int minR = Math.max(r - tolerance, 0)
    int minG = Math.max(g - tolerance, 0)
    int minB = Math.max(b - tolerance, 0)
    int maxR = Math.min(r + tolerance, 255)
    int maxG = Math.min(g + tolerance, 255)
    int maxB = Math.min(b + tolerance, 255)
    ImageProcessor ip = new ByteProcessor(image.getWidth(), image.getHeight());
//    def mask_img = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_BINARY)
    for (int i = 0; i < image.getWidth(); i++) {
        for (int j = 0; j < image.getHeight(); j++) {
            int color = image.getRGB(i, j)
            // (could use Java's Color class but this is probably a little faster)
            int red = (color >> 16) & 0x000000FF
            int green = (color >>8 ) & 0x000000FF
            int blue = (color) & 0x000000FF

//            def raster = mask_img.getRaster()

            if ( (red >= minR && red <= maxR) &&
                    (green >= minG && green <= maxG) &&
                    (blue >= minB && blue <= maxB) ) {
//                mask_img.setRGB(i, j, Color.WHITE.getRGB().byteValue())
                ip.putPixel(i, j, 255)
            }
            else
//                mask_img.setRGB(i, j, Color.BLACK.getRGB().byteValue())
                ip.putPixel(i, j, 0)
        }
    }
    return ip
}

print("DONE")










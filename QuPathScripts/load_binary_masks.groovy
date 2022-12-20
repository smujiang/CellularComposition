/**
 * * load binary mask images (usually from prediction) into QuPath, so as to have better visualization and interaction
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
import qupath.lib.objects.PathDetectionObject
import qupath.lib.roi.ROIs

//import qupath.imagej.objects.ROIConverterIJ
//import qupath.imagej.tools.IJTools
import qupath.lib.roi.RoiTools

//import qupath.lib.scripting.QP
import qupath.lib.gui.scripting.QPEx
import qupath.lib.objects.classes.PathClassFactory
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
// TODO: 2. each connected component as a cell (Detection in QuPath), not Annotation in QuPath
// TODO: 3. load class_txt_list, class_color_list from outside files;
//

// List<String> class_txt_list = ["Tumor", "Lymphocyte", "Stroma", "Non-Stroma"]
// List<List<Integer>> class_color_list = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 0, 0]]

def bkg = new Color(255,255,255) // white

def label_txt = "None"  // could be 'Tumor'... (any other in  QuPath class list)
//def label_txt = "Tumor"  // could be 'Tumor'... (any other in QuPath class list )

def Load_As = "Annotation"  // could also be "Detection"
//def Load_As = "Detection"

def hierarchy = QPEx.getCurrentHierarchy()

//img_fn = "/Users/My_LANID/Dataset/OvaryData/QBRC_all/512/OCMC-001_mask/OCMC-001_35972_33137.png"
//print(img_fn)
mask_dir = "/Users/My_LANID/Dataset/OvaryData/QBRC_all/512/OCMC-001_mask"
//mask_dir = "/Users/My_LANID/Dataset/OvaryData/Patches_out"
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
//        def all_annotations = parseMask(it, bkg, label_txt, Load_As)
        def annotations = []
        def img = ImageIO.read(it)

        // get original coordinate from image file name
        def end = it.name.lastIndexOf("_")
        def end_r = it.name.lastIndexOf(".png")
        def start = it.name.indexOf("_")
        char[] buf = new char[end - start-1]
        it.name.getChars(start+1, end, buf,0)
        int orig_x = Integer.parseInt(buf.toString())
        buf = new char[end_r - end - 1]
        it.name.getChars(end+1, end_r, buf,0)
        int orig_y = Integer.parseInt(buf.toString())

        // convert rgb "binary" image to TRUE binary image
        def ip = getColorSegmentation(img, bkg.red, bkg.green, bkg.blue, 0)
        def bi = ip.getBufferedImage()
        def imp = new ImagePlus(label_txt, bi)
        ManyBlobs allBlobs = new ManyBlobs(imp)
        allBlobs.setBackground(0)  // 0 for black, 1 for 255
        allBlobs.findConnectedComponents() // Start the Connected Component Algorithm
        for(int conn = 0; conn < allBlobs.size(); conn++){
            def x = allBlobs.get(conn).outerContour.xpoints
            def y = allBlobs.get(conn).outerContour.ypoints
            def tmp_points_list = []
            for(int i = 0; i < x.size(); i++){
                if(x[i] != 0 && y[i] != 0){
                    tmp_points_list.add(new Point2(x[i]+orig_x, y[i]+orig_y))
                }
            }
            def pathClass = null
            if (label_txt != 'None')
//                pathClass = PathClassFactory.getPathClass(label_txt)
                pathClass = label_txt

            def roi = new PolygonROI(tmp_points_list)
            if(Load_As == "Annotation"){
                def annotation = new PathAnnotationObject(roi)
                annotation.setPathClass(QPEx.getPathClass(pathClass))
                annotations.add(annotation)
            }
            else if(Load_As == "Detection"){
                def annotation = new PathDetectionObject(roi)
                annotation.setPathClass(QPEx.getPathClass(pathClass))
                annotations.add(annotation)
            }
            else{
                throw new Exception("Undefined mask loading option ")
            }
        }
        hierarchy.addPathObjects(annotations)
    } catch (Exception e) {
        print 'Unable to parse annotation from ' + it.getName() + ': ' + e.getLocalizedMessage()
    }
}

QPEx.fireHierarchyUpdate()

static def parseMask(File file, Color bkg, String label_txt, String Load_As){
    def annotations = []
    def img = ImageIO.read(file)

    // get original coordinate from image file name
    def end = file.name.lastIndexOf("_")
    def end_r = file.name.lastIndexOf(".png")
    def start = file.name.indexOf("_")
    char[] buf = new char[end - start-1]
    file.name.getChars(start+1, end, buf,0)
    int orig_x = Integer.parseInt(buf.toString())
    buf = new char[end_r - end - 1]
    file.name.getChars(end+1, end_r, buf,0)
    int orig_y = Integer.parseInt(buf.toString())

    // convert rgb "binary" image to TRUE binary image
    def ip = getColorSegmentation(img, bkg.red, bkg.green, bkg.blue, 0)
    def bi = ip.getBufferedImage()
    def imp = new ImagePlus(label_txt, bi)
    ManyBlobs allBlobs = new ManyBlobs(imp)
    allBlobs.setBackground(0)  // 0 for black, 1 for 255
    allBlobs.findConnectedComponents() // Start the Connected Component Algorithm
    for(int conn = 0; conn < allBlobs.size(); conn++){
        def x = allBlobs.get(conn).outerContour.xpoints
        def y = allBlobs.get(conn).outerContour.ypoints
        def tmp_points_list = []
        for(int i = 0; i < x.size(); i++){
            if(x[i] != 0 && y[i] != 0){
                tmp_points_list.add(new Point2(x[i]+orig_x, y[i]+orig_y))
            }
        }
        def pathClass = null
        if (label_txt != 'None'){
//            pathClass = PathClassFactory.getPathClass(label_txt)
            pathClass = label_txt
        }

        def roi = new PolygonROI(tmp_points_list)
        if(Load_As == "Annotation"){
            def annotation = new PathAnnotationObject(roi)
            annotation.setPathClass(QPEx.getPathClass(pathClass))
            annotations.add(annotation)
        }
        else if(Load_As == "Detection"){
            def annotation = new PathDetectionObject(roi)
            annotation.setPathClass(QPEx.getPathClass(pathClass))
            annotations.add(annotation)
        }
        else{
            throw new Exception("Undefined mask loading option ")
        }
    }
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

            if ( (red >= minR && red <= maxR) &&
                    (green >= minG && green <= maxG) &&
                    (blue >= minB && blue <= maxB) ) {
                ip.putPixel(i, j, 0) // for blob detection, background should be 0
            }
            else
                ip.putPixel(i, j, 255)
        }
    }
    return ip
}











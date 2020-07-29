/**
 * Script to export binary masks corresponding to all cells of an image as well as the image.
 * Note: 1. Region should be labeled as "ROI"
 *       2. Pay attention to the 'downsample' value to control the export resolution!
 * @author Jun Jiang (Jiang.Jun@mayo.edu), tested on QuPath 0.2.0-m8
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathCellObject
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP

//import qupath.lib.roi.PathROIToolsAwt  //if can't import this package, use the next line
import qupath.lib.roi.RoiTools

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = QP.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 1.0
// modify this root_dir to specify where you would like to save your export
//def pathOutput = "/Users/m192500/Dataset/Annotations/QP1.2_annotation/output"
def pathOutput = "H:\\OvarianCancer\\ForPatchClassification"

List<String> class_txt_list = ["Tumor", "Lymphocyte", "Stroma", "Macrophage", "Karyorrhexis", "RBC"]
List<List<Integer>> class_color_list = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 148, 255], [255, 0, 255]]

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: this just define the image extension, not mask extension. masks will always be exported as PNG
def imageExportType = 'JPG'

//def detections = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
def cell_detections = QP.getCellObjects()
for (anno in annotations){
    if(anno.getPathClass().toString().equals("ROI")){
        //TODO: get relevant cells in anno
//        saveAllCellMask(pathOutput, server, anno, cell_detections, downsample, imageExportType, false) //save cell (nucleus+plasma)
        saveAllCellMask(pathOutput, server, anno, cell_detections, downsample, imageExportType, true, class_txt_list, class_color_list) //save cell (nucleus)
    }
}

print 'Done!'

/**
 *
 * @param pathOutput: folder for output
 * @param server: QuPath Image server
 * @param ROI: ROI where the cells are located in
 * @param cell_detections: detected cells
 * @param downsample: downsample rate
 * @param imageExportType: output file extension
 * @param only_nucleus: if true, just show nucleus
 * @return
 */
def saveAllCellMask(String pathOutput, ImageServer server, PathObject ROI, List<PathObject>  cell_detections, double downsample, String imageExportType, boolean only_nucleus,
                    List<String> class_txt_list, List<List<Integer>> class_color_list) {
    ////////////////////////////create an image in ROI////////////////////////////////
    def patch_roi = ROI.getROI()
    def patch_region = RegionRequest.createInstance(server.getPath(), downsample, patch_roi)
    // Request the BufferedImage
    def org_img = server.readBufferedImage(patch_region)

    print(server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.')))
    // Create a name
    String name = String.format('%s_(%.2f,%d,%d,%d,%d)',
//            server.getShortServerName(),
            // if getShortServerName() doesn't work, try next line
            server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.')),
            patch_region.getDownsample(),
            patch_region.getX(),
            patch_region.getY(),
            patch_region.getWidth(),
            patch_region.getHeight()
    )
    // Create filename & export
    if (imageExportType != null) {
//        pathOutput = pathOutput + File.separator + server.getShortServerName().take(server.getShortServerName().lastIndexOf('.'))
        // if getShortServerName() doesn't work, try next line
        pathOutput = pathOutput + File.separator + server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))


        File newDir = new File(pathOutput)
        if (!newDir.exists()) {
            newDir.mkdirs()
        }
        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
        ImageIO.write(org_img, imageExportType, fileImage)
    }

    ///////////////////////////Create a mask with cells/////////////////////////////////
//    def mask_img = new BufferedImage(org_img.getWidth(), org_img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def mask_img = new BufferedImage(org_img.getWidth(), org_img.getHeight(), BufferedImage.TYPE_INT_RGB)
    def g2d = mask_img.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.fillRect(0, 0, mask_img.width, mask_img.height) //create white background
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-patch_region.getX(), -patch_region.getY())  //should not be in for loop, otherwise the drawing will be incomplete
    for (pathCellObject in cell_detections){
        // Extract ROI & classification name
        def cell_roi = pathCellObject.getROI()
        if (only_nucleus){
            cell_roi = pathCellObject.getNucleusROI()
        }

        def pathClass = pathCellObject.getPathClass()
        def classificationName = pathClass == null ? 'None' : pathClass.toString()
        if (cell_roi == null) {
            print 'Warning! No ROI for object ' + pathCellObject + ' - cannot export corresponding region & mask'
            return
        }

        // Create a mask using Java2D functionality
        // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
//        def shape = PathROIToolsAwt.getShape(cell_roi) //if can't import PathROIToolsAwt, try the next line
        def shape = RoiTools.getShape(cell_roi)

        def label = pathCellObject.getPathClass().toString()
        for(int class_idx = 0; class_idx < class_txt_list.size(); class_idx++){
            if (label.equals(class_txt_list[class_idx])){
                def rgb_tuple = class_color_list[class_idx]
                g2d.setColor(new Color(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]))
            }
        }
        //g2d.draw(shape)
        g2d.fill(shape)
    }
    g2d.dispose()
    // Export the mask
    def fileMask = new File(pathOutput, name + '-mask.png')
    ImageIO.write(mask_img, 'PNG', fileMask)

}





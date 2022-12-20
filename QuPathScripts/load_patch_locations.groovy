import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.roi.RectangleROI
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import static com.xlson.groovycsv.CsvParser.parseCsv

//csv = '/Users/My_LANID/Project/QuPathScripts/locations.csv'
//def data_dir = '/Users/My_LANID/Dataset/OvaryCancer/CellClassification/QuPathMeasurements_analysis'
//def data_dir = '/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification'
String data_dir = 'H:\\OvaryCancer\\PatchClassification'

Double patch_w = 512
Double patch_h = 512
String class_label = "errors"

def server = QP.getCurrentImageData().getServer()
case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
csv_fn = data_dir + File.separator + "misclassified_patches_loc.txt"
fh = new File(csv_fn)
/*
example:
case_id,c_x,c_y
ocmc-001,1000,1000
ocmc-002,1020,1020
ocmc-004,1040,1040
ocmc-001,1060,1060
* */

def csv_content = fh.getText('utf-8')
def data_iterator = parseCsv(csv_content, separator: ',', readFirstLine: false)

p_sz_h = server.getPixelCalibration().pixelHeightMicrons
p_sz_w = server.getPixelCalibration().pixelWidthMicrons

def imageData = QPEx.getCurrentImageData()
for (line in data_iterator) {
    def case_id_from_txt = line[0]
    def c_x = line[1] as Double
    def c_y = line[2] as Double
    if (case_id == case_id_from_txt) {
        def roi = new RectangleROI(c_x, c_y, patch_w, patch_h)
        def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass(class_label))
        imageData.getHierarchy().addPathObject(annotation, false)
    }
}
QPEx.fireHierarchyUpdate()
print("Done")






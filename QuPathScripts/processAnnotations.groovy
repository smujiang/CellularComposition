/*
 * Author: Jun Jiang (Jiang.Jun@mayo.edu), tested on QuPath 0.2.0-m8
 * This is the script for annotation processing, pipeline step 2.
 * If any package is missing, it will throw errors. You need to drag the missing jar onto the QuPath window, and you just need to do this only once. This step copies the jar into QuPath's jar directory
 *  --------------------------------------------------
 * 1). get the annotations before any processing, in case annotations were messed up.
 *      a. TODO:check cell detections, and get those have been annotated, save the cells[Ac1, Ac2, ...] with labels
        b. check annotations:
            b1. get ROIs
            b2. check regions, and get those have been annotated, save the regions [R1, R2, ...]
            b3. check points, and get those have been annotated, save the coordinates(px, py) and label
   2). cell detection
        a. select ROIs, rather than all the areas
        b. run cell detection in ROIs, and save all detected cells [C1, C2, ...]
   3). assign labels to cells
        a. loop through all cells and get the center of the cells (x, y)
        b. check if
           i. (x, y) in a region [R1, R2, ...]
           ii. TODO: (x, y) in a cell with annotation [Ac1, Ac2, ...]
           iii. (px, py) in a cell [C1, C2, ...]
           if true, assign the label to detected cells
   4). TODO: add those cells and fire Hierarchy update
 * */

import qupath.lib.roi.PointsROI
import qupath.lib.roi.PolygonROI
import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
import qupath.lib.objects.classes.PathClassFactory


def default_class = "Stroma"
// get pixel size and unit
def server = QP.getCurrentServer()
def cal = server.getPixelCalibration()
String xUnit = cal.getPixelWidthUnit()
String yUnit = cal.getPixelHeightUnit()
double pixelWidth = cal.getPixelWidthMicrons()
double pixelHeight = cal.getPixelHeightMicrons()
print("----------------------------------------------")
print("Processing " + server.toString())
// 1. get all the useful annotations
print("\t -Get manual annotations")
def hierarchy = QP.getCurrentHierarchy()
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def ROI_list = []
def Polygon_annotation_list = []
def Point_annotation_list = []
for (anno in annotations){
    if(anno.getPathClass().toString().equals("ROI")){
        ROI_list << anno
    }
    else if(anno.getROI() instanceof PointsROI){
        Point_annotation_list << anno
    }
    else if(anno.getROI() instanceof PolygonROI){ //TODO: important!!! could also be geometry
        Polygon_annotation_list << anno
    }
}
// 2. Select ROI and do cell detection
print("\t -Select ROIs and do cell detection")
QPEx.selectObjects(ROI_list)
//selectAnnotations(ROI_list)
QPEx.runPlugin('qupath.imagej.detect.cells.WatershedCellDetection',
        '{"requestedPixelSizeMicrons": 0.25, "backgroundRadiusMicrons": 8, "medianRadiusMicrons": 0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "watershedPostProcess": true,  "cellExpansionMicrons": 1.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

// 3. Assign labels to detected cells
print("\t -Assign labels to detected cells")
def detections = hierarchy.getFlattenedObjectList(null).findAll {it.isDetection()}
def updated_detections = []
int total_cnt = 1
int annotated_cnt = 1
for (d in detections){
    if (d.isCell()){
        print(total_cnt)
        if (! d.pathClass){ // not assigned a label yet
            cell_x = d.getROI().centroidX
            cell_y = d.getROI().centroidY
            for (polygon in Polygon_annotation_list){
                if (polygon.getROI().contains(cell_x, cell_y)){
                    println(sprintf('\t\t -Cell is labeled by a polygon: %f,%f(%s), Label: %s', cell_x*pixelWidth, cell_y*pixelHeight, xUnit, polygon.getPathClass().toString()))
                    d.setPathClass(polygon.getPathClass())
                    annotated_cnt += 1
                }
            }
            for (points in Point_annotation_list){
                for (p in points.ROI.allPoints){
                    if (d.getNucleusROI().contains(p.x, p.y)){
                        println(sprintf('\t\t -Cell is labeled by a point: %f,%f(%s), Label: %s', p.x*pixelWidth, p.y*pixelHeight, xUnit, points.getPathClass().toString()))
                        d.setPathClass(points.getPathClass())
                        annotated_cnt += 1
                    }
                }
            }
            updated_detections << d  // add the updated detection into a list
        }
        else {
            print("\t\t -Get previously labeled " + d.pathClass.toString())
            annotated_cnt += 1
        }
        total_cnt += 1
    }
}

for (d in detections){
    if (! d.pathClass){
        d.setPathClass(PathClassFactory.getPathClass(default_class))
    }
    updated_detections << d  // add the updated detection into a list
}
print(sprintf("\t\t -Annotated cell count in total: %d", annotated_cnt))

//4. Update QuPath view
// TODO: need to delete current detection and add updated detection?
for (d in updated_detections){
    QPEx.getCurrentImageData().getHierarchy().addPathObject(d, false)
}
QPEx.fireHierarchyUpdate()



print "finished"








import xml.etree.ElementTree as ET

classes = ["slide"]

def convert_annotation(image_id, list_file):
    in_file = open('LandSlideDataSet/Annotations/%s.xml' %(image_id),encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))

        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


image_ids = open('./LandSlideDataSet/ImageSets/train.txt').read().strip().split()
list_file = open('landslide_train.txt', 'w')
for image_id in image_ids:
    list_file.write('./LandSlideDataSet/images/%s.tif' % (image_id))
    convert_annotation(image_id, list_file)
    list_file.write('\n')
list_file.close()
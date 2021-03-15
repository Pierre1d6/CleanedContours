import os
import pandas as pd
from radiomics import featureextractor, setVerbosity
import six
import csv
from pathlib import Path
import SimpleITK as sitk
from termcolor import colored
import numpy as np


def PyradToCSV(results,fileName):
    keys, values = [], []
    for key, val in six.iteritems(results):
        # print("\t%s: %s" %(key, val))
        if key.startswith("original"):
            keys.append(key)
            values.append(val)

    with open(fileName, "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)


pathData = "Data5centers" #Data test
pathMask = "Data5centers" #"nii_train_original_split" test


pathParams = "Params"

maskType = "original"   # "hecktor" OR "original"
maskGtv = "_gtvt_"  # "_gtvt_" OR "_gtvn_"


pdPatient = pd.DataFrame()

setVerbosity(40)
flagPET = False
flagCT = False

for patient in os.listdir(pathMask):#os.listdir(pathMask):  # pathMask pour etre sur d'avoir les masques DEBUG : ["CHUM034"]
    print("patient ID : " + colored(patient,"red"))
    pathPatientImage = os.path.join(pathData,patient)
    pathPatientMask = os.path.join(pathMask,patient)
    pdPETCT = pd.DataFrame()
    pdCT = pd.DataFrame()
    pdPET = pd.DataFrame()

    for image in os.listdir(pathPatientImage):
        imageName , imageExtension = os.path.splitext(image)
        if "ct.nii.gz" in image:
            if not "_LOG_" in image:
                pathParamsCT = os.path.join(pathParams,"CT/Original")

                #pathParamsCT = os.path.join(pathParams,"CT")
                pathPatientImageCT = os.path.join(pathPatientImage,image)
                pathPatientMaskCT = os.path.join(pathPatientMask,patient+maskGtv+maskType+".nii.gz")
                if os.path.exists(pathPatientMaskCT):
                    ma = sitk.ReadImage(pathPatientMaskCT, sitk.sitkUInt8)
                    ccif = sitk.ConnectedComponentImageFilter()
                    ccif.SetFullyConnected(True)
                    new_ma = ccif.Execute(ma)
                    numNodulesCT = ccif.GetObjectCount()
                    print("# Connected components (CT) : " + colored(numNodulesCT, "green") + " Image id # "+colored(image,"red"))
                    for paramsCT in os.listdir(pathParamsCT):
                        ParamsCT = Path(paramsCT).stem[6:]
                        extractor = featureextractor.RadiomicsFeatureExtractor(os.path.join(pathParamsCT,paramsCT))
                        extractor.addProvenance(False)
                        results= pd.DataFrame()
                        for l in range(1,numNodulesCT + 1):
                            if np.sum(sitk.GetArrayFromImage(new_ma) == l) >= 20:
                                #print(str(l))
                                result = extractor.execute(pathPatientImageCT, new_ma,label=l)
                                df = pd.DataFrame(result, columns=result.keys(), index=[patient])
                                results = pd.concat([results,df],axis=0)

                        results = results.mean(axis=0,level=0)
                        #print(results)
                        df = pd.DataFrame(results, columns=results.keys(), index=[patient])
                        df = df.add_prefix(imageName[8:-4] + "_" + ParamsCT+"_")
                        pdCT = pd.concat([pdCT, df], axis=1)
                        flagCT = True

                else:
                    flagCT = False

        elif "pt.nii.gz" in image:
            imageName, imageExtension = os.path.splitext(image)
            if not "_LOG_" in image:
                pathParamsPET = os.path.join(pathParams, "PET/Original")

                pathPatientImagePET = os.path.join(pathPatientImage, image)
                pathPatientMaskPET = os.path.join(pathPatientMask, patient + maskGtv + maskType + ".nii.gz")
                if os.path.exists(pathPatientMaskPET):
                    ma = sitk.ReadImage(pathPatientMaskPET, sitk.sitkUInt8)
                    ccif = sitk.ConnectedComponentImageFilter()
                    ccif.SetFullyConnected(True)
                    new_ma = ccif.Execute(ma)
                    numNodulesPET = ccif.GetObjectCount()
                    print("# Connected components (PET) : " + colored(numNodulesPET, "green")+ " Image id # "+colored(image,"red"))
                    for paramsPET in os.listdir(pathParamsPET):
                        ParamsPET = Path(paramsPET).stem[6:]
                        #print(os.path.join(pathParamsPET, paramsPET))
                        #print(pathPatientImagePET)
                        extractor = featureextractor.RadiomicsFeatureExtractor(os.path.join(pathParamsPET, paramsPET))
                        extractor.addProvenance(False)
                        results = pd.DataFrame()
                        for l in range(1, numNodulesPET + 1):
                            if np.sum(sitk.GetArrayFromImage(new_ma)==l) >= 20 :
                                #print(str(l))
                                result = extractor.execute(pathPatientImagePET, new_ma, label=l)
                                df = pd.DataFrame(result, columns=result.keys(), index=[patient])
                                results = pd.concat([results, df], axis=0)
                        results = results.mean(axis=0, level=0)
                        df = pd.DataFrame(results, columns=results.keys(), index=[patient])
                        df = df.add_prefix(imageName[8:-4] + "_" + ParamsPET + "_")
                        pdPET = pd.concat([pdPET, df], axis=1)
                        flagPET = True


                else:
                    flagPET = False

    if flagCT and flagPET:
        Number = pd.DataFrame(data=[numNodulesCT],index=[patient],columns=["NumberNodules"])
        pdPETCT = pd.concat([Number,pdCT, pdPET], axis=1)
        #print(pdPETCT)
        pdPatient = pd.concat([pdPatient, pdPETCT])
        print(pdPatient.shape)
        pdPatient.to_csv(maskType + maskGtv + "FeaturesV3.csv")

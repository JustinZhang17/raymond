// External Imports
import { useState } from "react";
import { HiOutlineCloudUpload } from "react-icons/hi";
import Image from "next/image";
import { motion } from "framer-motion";
import * as tf from "@tensorflow/tfjs";

// Internal imports
import Button from "@/components/atoms/button";
import Placeholder from "@/assets/placeholder.jpg";

const getTumorType = (values: number[]): string => {
  const max = Math.max(...values);
  const index = values.indexOf(max);

  const prefix = `I'm ${(max * 100).toFixed(2)}% sure, That's`;
  if (index === 0) return `${prefix} a Glioma Tumor.`;
  if (index === 1) return `${prefix} a Menigioma Tumor.`;
  if (index === 2) return `${prefix} no Tumor.`;
  if (index === 3) return `${prefix} a Pituitary Tumor.`;
  return "Can't Identify Isopod";
};

const Identification = (): JSX.Element => {
  const [image, setImage] = useState(null);
  const [createObjectURL, setCreateObjectURL] = useState("");
  const [classification, setClassification] = useState(
    "Please Upload an Image!"
  );

  const identifyTumor = async (): Promise<void> => {
    if (!image) {
      alert("Please upload an image.");
      return;
    }

    setClassification("Classifying...");

    const imageEle = document.getElementById("uploaded-image");

    // TODO: Should be moved onto the server side, do more research on TFJS how to do this
    const model = await tf.loadLayersModel("/model/model.json");

    const loadedImage = tf.browser
      // @ts-ignore
      .fromPixels(imageEle, 3)
      .resizeBilinear([256, 256])
      .toFloat()
      .mul(1.0 / 255.0)
      .expandDims(0);

    const prediction = model.predict(loadedImage);

    // @ts-ignore
    const values = prediction.dataSync();

    setClassification(getTumorType(values));
  };

  // TODO: Find Type for e
  const uploadToClient = (e: any): void => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      if (file.type !== "image/png" && file.type !== "image/jpeg") {
        alert("Only PNG and JPG files are supported.");
        return;
      }
      setImage(file);
      const path = URL.createObjectURL(file);
      setCreateObjectURL(path);
      setClassification("Ready to Identify!");
    }
  };
  return (
    <section className="mx-4 sm:mx-0">
      <motion.h1
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="py-12 px-0 text-3xl font-extrabold md:px-2 md:text-5xl xl:text-6xl 2xl:text-7xl text-center"
      >
        MRI Brain Scan Identifier
      </motion.h1>
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        whileInView={{
          y: 0,
          opacity: 1,
          transition: { duration: 0.7, ease: "easeOut" },
        }}
        className="flex justify-center gap-4 md:flex-row flex-col"
      >
        <div className="flex flex-col items-center justify-center gap-4 ">
          <motion.div
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            <Image
              id="uploaded-image"
              src={createObjectURL ? createObjectURL : Placeholder}
              width={400}
              height={400}
              alt="Uploaded Image"
              className="rounded-lg max-h-96 object-cover"
            />
          </motion.div>
          <div className="text-xl font-extrabold">
            {classification === "Classifying..." ? (
              <span className=" animate-ping">Loading</span>
            ) : (
              classification
            )}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-6 ">
          <label
            aria-label="upload file"
            htmlFor="dropzone-file"
            className="flex p-12 bg-white flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-background-light dark:bg-background-dark dark:hover:bg-text-dark  hover:bg-text-light dark:border-gray-600  "
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <HiOutlineCloudUpload className="w-10 h-10 mb-3 text-default-dark dark:text-default-light" />
              <p className="mb-2 text-sm text-default-dark dark:text-default-light">
                <span className="font-semibold">Click or Drag&nbsp;</span>
                To Upload
              </p>
              <p className="text-xs text-default-dark dark:text-default-light">
                PNG or JPG
              </p>
            </div>
            <input
              id="dropzone-file"
              type="file"
              className="hidden"
              onChange={uploadToClient}
            />
          </label>
          <div className="flex justify-center md:mb-0">
            <Button onClick={identifyTumor}>Submit</Button>
          </div>
        </div>
      </motion.div>
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="py-12 px-0 font-extrabold md:px-2 text-sm text-center"
      >
        This is a project created for educational purposes only. The model is
        not intended to be used in a clinical setting. <br /> It only classifies
        images into 1 of 4 categories. Glioma Tumor, Meningioma Tumor, Pituitary
        Tumor, and No Tumor.
      </motion.div>
    </section>
  );
};

export default Identification;

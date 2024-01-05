# https://github.com/karolzak/keras-unet
import json
import math

# import os

import cv2
import numpy as np
import tensorflow as tf
from keras_unet.models import custom_unet
from PIL import Image

from .Patch import PatchedImage
from .util import (
    crop_to_bounds,
    expand2square,
    find_center_mask,
    get_actual_img_bounds,
    read_PIL_image,
)


class UnetSpotExtractor:
    def __init__(
        self, inputShape=(512, 512, 3), modelWeights=None, dropout=0.05, useGPU=True
    ):
        self.inputShape = inputShape
        self.model = custom_unet(
            input_shape=inputShape,
            use_batch_norm=False,
            filters=64,
            dropout=dropout,
            num_layers=4,
            output_activation="sigmoid",
        )
        self.useGPU = useGPU
        if modelWeights is not None:
            try:
                self.model.load_weights(modelWeights)
                self.trained = True
            except IOError as e:
                print(e)
                self.trained = False
        else:
            self.trained = False

    def loadWeights(self, weightFile):
        self.model.load_weights(weightFile)

    def predict(self, images, batchSize=5):
        deviceToUse = "/device:CPU:0"
        if self.useGPU:
            deviceToUse = "/device:GPU:0"
        if self.trained:
            batches = np.array_split(images, math.ceil(len(images) / batchSize))
            print(len(images), batchSize, len(batches))
            results = []
            with tf.device(deviceToUse):
                for batch in batches:
                    results.append(self.model.predict(batch))
            return np.concatenate(results, axis=0)
        else:
            print("Model not trained")
            return images

    def generate_spots(
        self,
        fishes,
        maskPaths,
        batch_size=5,
        verbose=False,
        threshold=-1,
    ):
        """
        Generates spots for the given fishes using the trained model.

        Args:
            fishes (list): A list of Fish objects.
            maskPaths (list): A list of file paths to the corresponding mask images.
            batch_size (int, optional): The batch size for processing the images. Defaults to 5.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            threshold (int, optional): The threshold value for binarizing the predicted spots.
                Pixels with values greater than the threshold will be set to 1, and pixels with
                values less than or equal to the threshold will be set to 0. Defaults to -1.

        Returns:
            None
        """
        if self.trained:
            batchNum = 1
            batches = np.array_split(fishes, math.ceil(len(fishes) / batch_size))
            batches_masks = np.array_split(
                maskPaths, math.ceil(len(fishes) / batch_size)
            )
            for batchNum, batch in enumerate(batches):
                if verbose:
                    print(f"Batch: {batchNum + 1}/{len(batches_masks)}")
                imgs_list = []
                i = 0
                for fish in batch:
                    # Process each fish image
                    img = read_PIL_image(fish.image_path)
                    # If input to model is grayscale, convert to grayscale
                    if self.inputShape[2] == 1:
                        img = img.convert("L")

                    if fish.mask_path is None:
                        if verbose:
                            print(f"No mask available for: {fish.uuid}")
                        continue
                    fm = read_PIL_image(fish.mask_path).convert("L")

                    if len(np.where(np.asarray(fm).max(axis=0) > 0)[0]) <= 0:
                        if verbose:
                            print(f"No mask present in: {fish.mask_path}")
                        continue

                    maskBounds = get_actual_img_bounds(fm)

                    colorMode = "L" if self.inputShape[2] == 1 else "RGB"
                    img_masked = Image.new(colorMode, img.size)
                    img_masked.paste(img, mask=fm)
                    img_masked = crop_to_bounds(img_masked, maskBounds)

                    originalShape = img_masked.size

                    ratio = img_masked.size[1] / img_masked.size[0]
                    img_masked = expand2square(
                        img_masked.resize(
                            (self.inputShape[0], round(self.inputShape[0] * ratio))
                        )
                    )

                    if verbose:
                        print(i, fish.image_path, img_masked.size, originalShape)
                    if (
                        img_masked.size[0] != self.inputShape[0]
                        or img_masked.size[1] != self.inputShape[1]
                    ):
                        if verbose:
                            print("Something went wrong: Invalid masked image size.")
                    else:
                        imgs_list.append(
                            (
                                np.array(img_masked),
                                fish,
                                ratio,
                                originalShape,
                            )
                        )
                    i += 1
                if verbose:
                    print("Predicting...")
                if len(imgs_list) > 0:
                    only_img_list = np.asarray([x[0] for x in imgs_list])
                    spots = self.predict(only_img_list, batchSize=batch_size)
                    if threshold >= 0:
                        spots[spots > threshold] = 1
                        spots[spots <= threshold] = 0

                    imgs_list = [
                        (
                            imgs_list[i][0],
                            imgs_list[i][1],
                            imgs_list[i][2],
                            imgs_list[i][3],
                            spots[i],
                        )
                        for i in range(len(imgs_list))
                    ]
                    if verbose:
                        print("Saving...")

                    for img, fish, ratio, originalShape, spots in imgs_list:
                        # Figure out right height
                        currentSpotsHeight = ratio * spots.shape[1]
                        spots = spots.reshape(spots.shape[0], spots.shape[1])
                        newSpots = crop_to_bounds(
                            spots,
                            [
                                round(spots.shape[1] / 2 - currentSpotsHeight / 2),
                                round(spots.shape[1] / 2 + currentSpotsHeight / 2),
                                0,
                                spots.shape[0],
                            ],
                        )
                        newSpots = np.asarray(newSpots.resize(originalShape))
                        newSpots = np.asarray(newSpots * 255, dtype=np.uint8)

                        if verbose:
                            print(fish.spot_path)
                        Image.fromarray(newSpots).save(fish.spot_path)
                        with open(fish.spotJson, "w", encoding="utf-8") as f:
                            json.dump(getPointsFromImage(newSpots), f)
                else:
                    print("No images in batch to process...")

                batchNum += 1
        else:
            print("Network not trained")

    def generate_spots_patched(
        self,
        fishes,
        batch_size=5,
        verbose=False,
        threshold=-1,
    ):
        """
        Generates filled-in spots for the given fishes using the trained model.

        Args:
            fishes (list): A list of Fish objects.
            batch_size (int, optional): The batch size for processing the images. Defaults to 5.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            threshold (int, optional): The threshold value for binarizing the predicted spots.
                Pixels with values greater than the threshold will be set to 1, and pixels with
                values less than or equal to the threshold will be set to 0. Defaults to -1.

        Returns:
            None
        """
        if self.trained:
            img_num = 0
            for fish in fishes:
                if verbose:
                    print(f"({img_num + 1}/{len(fishes)}) Image: {fish.image_path}")

                img = read_PIL_image(fish.image_path)
                # If input to model is grayscale, convert to grayscale
                if self.inputShape[2] == 1:
                    img = img.convert("L")

                if fish.mask_path is None:
                    if verbose:
                        print(f"No mask avaliable for: {fish.image_path}")
                    continue
                fm = read_PIL_image(fish.mask_path).convert("L")

                if len(np.where(np.asarray(fm).max(axis=0) > 0)[0]) <= 0:
                    if verbose:
                        print(f"No mask present in: {fish.mask_path}")
                    continue

                maskBounds = get_actual_img_bounds(fm)

                colorMode = "L" if self.inputShape[2] == 1 else "RGB"
                img_masked = Image.new(colorMode, img.size)
                img_masked.paste(img, mask=fm)
                img_masked = crop_to_bounds(img_masked, maskBounds)

                patchedImage = PatchedImage(
                    (self.inputShape[0], self.inputShape[1]),
                    min(self.inputShape[0], self.inputShape[1]) / 4,
                    img_masked,
                )

                if verbose:
                    print("Predicting...", img_masked.size, patchedImage.size)
                print(patchedImage.asarray(dtype=np.float32).shape)

                spots = self.predict(
                    patchedImage.asarray(dtype=np.float32) / 255, batchSize=batch_size
                )

                if verbose:
                    print("Saving...")

                if threshold >= 0:
                    spots[spots > threshold] = 1
                    spots[spots <= threshold] = 0

                print(spots.shape)
                patchedImage.fromarray((spots * 255).astype(np.uint8))

                spotsFullImg = patchedImage.joinPatches()
                print(spotsFullImg.size)

                if verbose:
                    print(fish.spot_path)
                spotsFullImg.save(fish.spot_path)
                with open(fish.spotJson, "w", encoding="utf-8") as f:
                    json.dump(getPointsFromImage(np.asarray(spotsFullImg)), f)

                img_num += 1
        else:
            print("Network not trained")


class UnetMaskExtractor:
    def __init__(
        self, inputShape=(512, 512, 3), modelWeights=None, useGPU=True, dropout=0.05
    ):
        self.inputShape = inputShape
        self.model = custom_unet(
            input_shape=inputShape,
            use_batch_norm=False,
            filters=64,
            dropout=dropout,
            num_layers=4,
            output_activation="sigmoid",
        )
        self.useGPU = useGPU
        if modelWeights is not None:
            try:
                self.model.load_weights(modelWeights)
                self.trained = True
            except IOError as e:
                print(e)
                self.trained = False
        else:
            self.trained = False

    def loadWeights(self, weightFile):
        self.model.load_weights(weightFile)

    def predict(self, images, batchSize=5, postProcess=True, threshold=0.31):
        deviceToUse = "/device:CPU:0"
        if self.useGPU:
            deviceToUse = "/device:GPU:0"
        if self.trained:
            batches = np.array_split(images, math.ceil(len(images) / batchSize))
            results = []
            with tf.device(deviceToUse):
                for batch in batches:
                    if postProcess:
                        results.append(
                            self.postProcessPredictions(
                                self.model.predict(batch), threshold=threshold
                            )
                        )
                    else:
                        results.append(self.model.predict(batch))

            return np.concatenate(results, axis=0)
        else:
            print("Model not trained")
            return images

    def postProcessPredictions(
        self,
        predictions,
        threshold=0.31,
        kernel_size=(3, 3),
        iterations=3,
        min_mask_area=100,
    ):
        predictions = predictions.reshape(
            predictions.shape[0], predictions.shape[1], predictions.shape[2]
        )
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        kernel = np.ones(kernel_size, np.uint8)
        dilated_images = []
        for img in predictions:
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = img.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            im_floodfill = img.copy()
            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0, 0), 1)

            # Invert floodfilled image
            im_floodfill_inv = 1 - im_floodfill
            # Combine the two images to get the foreground.
            im_out = np.bitwise_or(
                img.astype(np.uint8), im_floodfill_inv.astype(np.uint8)
            )

            # Apply erosion followed by expansion to merge up separated predictions
            dilated = cv2.morphologyEx(
                im_out, cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
            dilated_images.append(
                find_center_mask(
                    np.asarray(dilated * 255, dtype=np.uint8),
                    min_mask_area=min_mask_area,
                )
            )

        return np.stack(dilated_images, axis=0)

    def prepareInput(self, imgs, verbose=False):
        imgs_list = []
        i = 0
        for image in imgs:
            img = read_PIL_image(image)
            # If input to model is grayscale, convert to grayscale
            if self.inputShape[2] == 1:
                img = img.convert("L")

            ratio = img.size[1] / img.size[0]
            img = expand2square(
                img.resize((self.inputShape[0], round(self.inputShape[0] * ratio)))
            )
            if verbose:
                print(i, image, img.size)
            imgs_list.append(np.array(img))
            i += 1
        return imgs_list

    def generate_masks(
        self,
        fish,
        batch_size=5,
        verbose=False,
        threshold=0.31,
    ):
        """
        Generates masks for a batch of fish images.

        Args:
            fish (list): List of fish objects.
            batch_size (int, optional): Number of fish images to process in each batch. Defaults to 5.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            threshold (float, optional): Threshold value for mask prediction. Defaults to 0.31.

        Returns:
            None
        """
        if self.trained:
            batchNum = 1
            batches = np.array_split(fish, math.ceil(len(fish) / batch_size))
            for batch in batches:
                if verbose:
                    print(f"Batch: {batchNum}/{len(batches)}")
                imgs_list = []
                i = 0
                for fish in batch:
                    origImg = read_PIL_image(fish.image_path)
                    originalShape = origImg.size
                    # If input to model is grayscale, convert to grayscale
                    if self.inputShape[2] == 1:
                        origImg = origImg.convert("L")
                    ratio = origImg.size[1] / origImg.size[0]
                    tmp = origImg.resize(
                        (self.inputShape[0], round(self.inputShape[0] * ratio))
                    )
                    img = expand2square(tmp)
                    if verbose:
                        print(i, fish.uuid, originalShape, img.size, tmp.size)
                        i += 1
                    imgs_list.append(
                        (
                            np.array(img),
                            ratio,
                            originalShape,
                            fish,
                            np.asarray(origImg),
                        )
                    )
                if verbose:
                    print("Predicting...")
                only_img_list = np.asarray([x[0] for x in imgs_list])
                masks = self.predict(
                    only_img_list, batchSize=batch_size, threshold=threshold
                )

                imgs_list = [
                    (
                        imgs_list[i][0],
                        imgs_list[i][1],
                        imgs_list[i][2],
                        imgs_list[i][3],
                        imgs_list[i][4],
                        masks[i],
                    )
                    for i in range(len(imgs_list))
                ]
                if verbose:
                    print("Saving...")

                for img, ratio, originalShape, fish, _, mask in imgs_list:
                    # Figure out right height
                    currentMaskHeight = ratio * mask.shape[1]
                    newMask = crop_to_bounds(
                        mask,
                        [
                            round(mask.shape[1] / 2 - currentMaskHeight / 2),
                            round(mask.shape[1] / 2 + currentMaskHeight / 2),
                            0,
                            mask.shape[0],
                        ],
                    )
                    newMask = np.asarray(newMask.resize(originalShape))
                    if len(np.where(newMask.max(axis=0) > 0)[0]) <= 0:
                        if verbose:
                            print(
                                f"Could not extract fish mask succesfully from: {fish.uuid}"
                            )
                    Image.fromarray(newMask).save(fish.mask_path)
                    batchNum += 1
        else:
            print("Network not trained")


def getPointsFromImage(img):
    contour, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    spotsDrawn = 0
    points = []
    for cnt in sorted(contour, key=cv2.contourArea, reverse=True):
        M = cv2.moments(cnt)
        cx = 0
        cy = 0
        if M["m00"] == 0:
            cx = int(cnt[0][0][0])
            cy = int(cnt[0][0][1])
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        radius = cv2.minEnclosingCircle(cnt)
        points.append((cx, cy, radius))
        spotsDrawn += 1
    return points

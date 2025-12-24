package com.ibrahimcanerdogan.facedetection.camera

import android.graphics.Rect
import android.media.MediaPlayer
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ibrahimcanerdogan.facedetection.R
import com.ibrahimcanerdogan.facedetection.graphic.GraphicOverlay
import com.ibrahimcanerdogan.facedetection.graphic.RectangleOverlay

class CameraAnalyzer(
    private val overlay: GraphicOverlay<*>
) : BaseCameraAnalyzer<List<Face>>() {

    override val graphicOverlay: GraphicOverlay<*>
        get() = overlay

    private val cameraOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.15f)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(cameraOptions)


    private val EYE_CLOSE_THRESHOLD = 0.3f
    private var mediaPlayer: MediaPlayer? = null
    private var eyesClosedStartTime: Long = 0
    private val MIN_CLOSED_DURATION = 1500L

    override fun detectInImage(image: InputImage): Task<List<Face>> {
        return detector.process(image)
    }

    override fun stop() {
        try {
            detector.close()
            mediaPlayer?.stop()
            mediaPlayer?.release()
            mediaPlayer = null
        } catch (e: Exception) {
            Log.e(TAG, "stop: $e")
        }
    }

    override fun onSuccess(
        results: List<Face>,
        graphicOverlay: GraphicOverlay<*>,
        rect: Rect
    ) {
        graphicOverlay.clear()

        results.forEach { face ->
            handleAlarm(face.leftEyeOpenProbability, face.rightEyeOpenProbability)
            val faceGraphic = RectangleOverlay(graphicOverlay, face, rect)
            graphicOverlay.add(faceGraphic)
        }

        graphicOverlay.postInvalidate()
    }

    private fun handleAlarm(leftEye: Float?, rightEye: Float?) {
        if (leftEye != null && rightEye != null) {
            val eyesClosed = leftEye < EYE_CLOSE_THRESHOLD && rightEye < EYE_CLOSE_THRESHOLD
            val currentTime = System.currentTimeMillis()

            if (eyesClosed) {
                if (eyesClosedStartTime == 0L) {

                    eyesClosedStartTime = currentTime
                } else if (currentTime - eyesClosedStartTime >= MIN_CLOSED_DURATION) {

                    if (mediaPlayer == null) {
                        mediaPlayer = MediaPlayer.create(
                            graphicOverlay.context,
                            R.raw.alarm_sound
                        )
                        mediaPlayer?.isLooping = true
                        mediaPlayer?.start()
                    }
                }
            } else {

                mediaPlayer?.stop()
                mediaPlayer?.release()
                mediaPlayer = null
                eyesClosedStartTime = 0L
            }
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "onFailure : $e")
    }

    companion object {
        private const val TAG = "CameraAnalyzer"
    }
}

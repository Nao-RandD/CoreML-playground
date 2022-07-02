//
//  DigitClassifier.swift
//  CoreMLMnist
//
//  Created by Naoyuki Kan on 2022/07/02.
//

import Foundation
import Vision

class DigitClassifier {

    private let model: VNCoreMLModel

    init() {
        model = try! VNCoreMLModel(for: ModifiedMNISTDigitClassifier().model)
    }

    func run(inputImage: CGImage, completion: @escaping ([VNClassificationObservation]) -> Void) {
        let request = VNCoreMLRequest(model: model, completionHandler: { request, error in
            print("モデルの推定")
            guard let observations = request.results as? [VNClassificationObservation]
                else { fatalError() }
            completion(observations)
        })
        let handler = VNImageRequestHandler(cgImage: inputImage)
        try! handler.perform([request])
    }
}

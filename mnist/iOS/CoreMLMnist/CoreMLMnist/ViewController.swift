//
//  ViewController.swift
//  CoreMLMnist
//
//  Created by Naoyuki Kan on 2022/07/02.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    @IBOutlet private weak var drawView: DrawView!
    @IBOutlet private weak var predictionLabel: UILabel!
    @IBOutlet private weak var clearBtn: UIButton!

    private let classifier = DigitClassifier()

    override func viewDidLoad() {
        super.viewDidLoad()

        clearBtn.isHidden = true
        predictionLabel.text = nil
    }

    @IBAction func clearBtnTapped(sender: UIButton) {
        // clear the drawView
        drawView.lines = []
        drawView.setNeedsDisplay()
        predictionLabel.text = nil
        clearBtn.isHidden = true
    }

    @IBAction func detectBtnTapped(sender: UIButton) {
        // get the drawView context so we can get the pixel values from it to intput to network
        guard let cgImage = drawView.getViewContext()?.makeImage() else {return}

        print("画像の情報\(cgImage.height)×\(cgImage.width)、\(cgImage.utType)")

        classifier.run(inputImage: cgImage) { [weak self] observations in
            guard let self = self else { return }
            // show the prediction
            guard let firstResult = observations.first?.identifier else { return }
            self.predictionLabel.text = "\(firstResult)"
            self.clearBtn.isHidden = false
        }
    }
}


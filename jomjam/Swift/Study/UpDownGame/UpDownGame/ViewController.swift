//
//  ViewController.swift
//  UpDownGame
//
//  Created by 조재민 on 2021/06/10.
//

import UIKit

class ViewController: UIViewController {
    
    var randomValue: Int = 0
    var tryCount: Int = 0
    @IBOutlet weak var slider: UISlider!
    @IBOutlet weak var tryCountLabel: UILabel!
    @IBOutlet weak var sliderValueLabel: UILabel!
    @IBOutlet weak var minimumValueLabel: UILabel!
    @IBOutlet weak var maximumValueLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        self.reset()
    }
    
    @IBAction func sliderValueChanged(_ sender: UISlider) {
        let integerValue: Int = Int(sender.value)
        sliderValueLabel.text = String(integerValue)
    }
    
    func showAlert(message: String) {
        let alert = UIAlertController(title: nil, message: message, preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "OK", style: .default) { action in
            self.reset()
        }
        
        alert.addAction(okAction)
        present(alert, animated: true, completion: nil)
    }
    
    @IBAction func tochUpHitButton(_ sender: UIButton){
        print(slider.value)
        let hitValue: Int = Int(slider.value)
        slider.value = Float(hitValue)
        
        tryCount = tryCount + 1
        tryCountLabel.text = "\(tryCount)/5"
        
        if randomValue == hitValue {
            //print("YOU HIT")
            self.showAlert(message: "YOU HIT")
            self.reset()
        } else if tryCount >= 5 {
            //print("YOU LOSE...")
            self.showAlert(message: "YOU LOSE...")
            reset()
        } else if randomValue > hitValue {
            slider.minimumValue = Float(hitValue)
            minimumValueLabel.text = String(hitValue)
        } else {
            slider.maximumValue = Float(hitValue)
            maximumValueLabel.text = String(hitValue)
        }
    }
    
    @IBAction func touchUpResetButton(_ sender: UIButton) {
        print("touch up reset button")
        self.reset()
    }
    
    func reset() {
        print("Reset!")
        randomValue = Int.random(in: 0...30)
        print(randomValue)
        tryCount = 0
        tryCountLabel.text = "0/5"
        slider.minimumValue = 0
        slider.maximumValue = 30
        slider.value = 15
        minimumValueLabel.text = "0"
        maximumValueLabel.text = "30"
        sliderValueLabel.text = "15"
    }

}


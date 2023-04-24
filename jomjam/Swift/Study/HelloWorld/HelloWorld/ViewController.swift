//
//  ViewController.swift
//  HelloWorld
//
//  Created by 조재민 on 2021/06/08.
//

import UIKit

class ViewController: UIViewController {

    @IBAction func Click_moveBtn(_ sender: Any) {
        //At storyboard find controller
        if let controller = self.storyboard?.instantiateViewController(withIdentifier: "DetailController"){
            
            //Push controller
            self.navigationController?.pushViewController(controller, animated: true)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
}


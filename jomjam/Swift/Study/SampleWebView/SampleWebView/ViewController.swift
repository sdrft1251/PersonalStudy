//
//  ViewController.swift
//  SampleWebView
//
//  Created by 조재민 on 2021/06/08.
//

import UIKit
import WebKit

class ViewController: UIViewController {

    @IBOutlet weak var WebViewMain: WKWebView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // 1. Get URL String
        // 2. Request URL
        // 3. req > load
        
        let urlString = "https://www.google.com"
        if let url = URL(string: urlString){
            //Unwrapping
            let urlReq = URLRequest(url: url)
            WebViewMain.load(urlReq)
        }
        
        
    }


}


//
//  ViewController.swift
//  TableView
//
//  Created by 조재민 on 2021/06/08.
//

import UIKit

class ViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    
    @IBOutlet weak var TableViewMain: UITableView!
    
    var newsData: Array<Dictionary<String, Any>>?
    
    func getNews() {
        let task = URLSession.shared.dataTask(with: URL(string: "https://newsapi.org/v2/top-headlines?country=kr&apiKey=21b0f90581244b44bf1538f6a8e715e9")!) { data, response, error in
            if let dataJson = data {
                do {
                    let json = try JSONSerialization.jsonObject(with: dataJson, options: []) as! Dictionary<String, Any>
                    let articles = json["articles"] as! Array<Dictionary<String, Any>>
                    self.newsData = articles
                    
                    DispatchQueue.main.sync{
                        self.TableViewMain.reloadData()
                    }
                }
                catch {}
                
            }
        }
        task.resume()
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        // 데이터 개수, 숫자 데이터
        
        if let news = newsData{
            return news.count
        } else {
            return 0
        }
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        // 어떤 데이터인지, 반복 10번?
        // 셀 사용법은 2가지
        //let cell = UITableViewCell.init(style: .default, reuseIdentifier: "TableCellType1")
        let cell = TableViewMain.dequeueReusableCell(withIdentifier: "Type1", for: indexPath) as! Type1
        
        let idx = indexPath.row
        if let news = newsData {
            
            let row = news[idx]
            if let r = row as? Dictionary <String, Any> {
                if let title = r["title"] as? String{
                    cell.LabelText.text = title
                }
            }
        }
        //cell.textLabel?.text = "\(indexPath.row)"
        //cell.LabelText.text = "\(indexPath.row)"
        
        return cell
    }
    
    //클릭 방법
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        print("CLICK !!!! \(indexPath.row)")
        
        let storyboard = UIStoryboard.init(name: "Main", bundle: nil)
        let controller = storyboard.instantiateViewController(identifier: "NewsDetailController") as! NewsDetailController
        
        if let news = newsData{
            let row = news[indexPath.row]
            if let r = row as? Dictionary<String, Any> {
                
                if let imageUrl = r["urlToImage"] as? String{
                    controller.imageUrl = imageUrl
                }
                if let desc = r["description"] as? String{
                    controller.desc = desc
                }
            }
        }
        // 이동이 수동
//        showDetailViewController(controller, sender: nil)
    }
    
    // 세그웨이 방법
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if let id = segue.identifier, "NewsDetail" == id {
            if let controller = segue.destination as? NewsDetailController {
                
                if let news = newsData{
                    if let indexPath = TableViewMain.indexPathForSelectedRow {
                        let row = news[indexPath.row]
                        if let r = row as? Dictionary<String, Any> {
                            if let imageUrl = r["urlToImage"] as? String {
                                controller.imageUrl = imageUrl
                            }
                            if let desc = r["description"] as? String {
                                controller.desc = desc
                            }
                        }
                    }
                    
                }
            }
        }
    // 이동이 자동
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        TableViewMain.delegate = self
        TableViewMain.dataSource = self
        
        getNews()
    }
    
    //tableview

}


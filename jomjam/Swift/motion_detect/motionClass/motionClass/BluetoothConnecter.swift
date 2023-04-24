//
//  BluetoothConnecter.swift
//  motionClass
//
//  Created by 조재민 on 2021/06/11.
//
import UIKit
import CoreBluetooth

class BluetoothConnecter: UIViewController {
    
    var centralManager: CBCentralManager!
    var spatchDevice: CBPeripheral!
    var nameString = "S-Patch EX "
    
    //let heartRateMeasurementCharacteristicCBUUID = CBUUID(string: "2A37")
    //let bodySensorLocationCharacteristicCBUUID = CBUUID(string: "2A38")
    
    let ecgServiceCBUUID = CBUUID(string: "66900001-da64-5a97-8c4f-04b8593ff99b")
    let imuCharacteristicCBUUID = CBUUID(string: "66900006-da64-5a97-8c4f-04b8593ff99b")
    let writeCharacteristicCBUUID = CBUUID(string: "66900002-da64-5a97-8c4f-04b8593ff99b")
    
    @IBOutlet weak var deviceNumber: UITextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    @IBAction func connectingBtn(_ sender: UIButton) {
        if deviceNumber.hasText == true, centralManager.state == CBManagerState.poweredOn {
            print("try to connect")
            if let deviceNumberString: String = deviceNumber.text {
                print(deviceNumberString)
                let zeroNeed: Int = 6 - deviceNumberString.count
                for _ in 1...zeroNeed {
                    nameString += "0"
                }
                nameString += deviceNumberString
                print("Try to connect " + nameString)
                centralManager.scanForPeripherals(withServices: nil)
            }
        } else {
            print("Need device number or Bluetooth is off")
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        if nameString.count == 17 && peripheral.name == nameString {
            print("FIND!!!")
            spatchDevice = peripheral
            spatchDevice.delegate = self
            centralManager.stopScan()
            centralManager.connect(spatchDevice)
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("Connected!!!")
        spatchDevice.discoverServices([ecgServiceCBUUID])
    }
    
}

extension BluetoothConnecter: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        
        case .unknown:
            print("jam!! unknown")
        case .resetting:
            print("jam!! resetting")
        case .unsupported:
            print("jam!! unsupported")
        case .unauthorized:
            print("jam!! unauthorized")
        case .poweredOff:
            print("jam!! poweredOff")
        case .poweredOn:
            print("jam!! poweredOn")
        @unknown default:
            print("jam!! default")
        }
    }
}

extension BluetoothConnecter: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let sevices = peripheral.services else { return }
        
        for service in sevices {
            print(service)
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }
    
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        
        var afterConnect = 0
        let signInBytes: [UInt8] = [0x00, 0x11, 0x11, 0x11, 0x11]
        let signInBytesData = NSData(bytes: signInBytes, length: signInBytes.count)
        
        let imuStartByte: [UInt8] = [0x0C, 0x11, 0x11, 0x11, 0x11, 0x03]
        let imuStartByteData = NSData(bytes: imuStartByte, length: imuStartByte.count)
        
        for characteristic in characteristics {
            if characteristic.uuid == writeCharacteristicCBUUID && afterConnect == 0 {
                print(characteristic)
                peripheral.writeValue(signInBytesData as Data, for: characteristic, type: CBCharacteristicWriteType.withResponse)
//                peripheral.readValue(for: characteristic)
//                peripheral.setNotifyValue(true, for: characteristic)
                afterConnect = 1
//                peripheral.discoverCharacteristics([imuCharacteristicCBUUID], for: service)
                peripheral.writeValue(imuStartByteData as Data, for: characteristic, type: CBCharacteristicWriteType.withResponse)
                
            }
            if characteristic.uuid == imuCharacteristicCBUUID && afterConnect == 1 {
                print("HERE")
                peripheral.setNotifyValue(true, for: characteristic)
//                print("IMU Characteristic found: \(characteristic.uuid)")
//                let imuMonitoringStart: [UInt8] = [12, 17, 17, 17, 17, 3, 0, 0, 0, 0, 0, 0, 0, 0]
//                let data = NSData(bytes: imuMonitoringStart, length: imuMonitoringStart.count)
//                print(data)
//                peripheral.writeValue(data as Data, for: characteristic, type: CBCharacteristicWriteType.withResponse)
//                peripheral.readValue(for: characteristic)
//                print("Read finish")
//                peripheral.setNotifyValue(true, for: characteristic)
//                print("Notify Finish")
//                let baseString = convertToBase64(hex: "0c11111111030000000000000000")
//                print(baseString)
//                let data = Data(base64Encoded: baseString)!
//                peripheral.writeValue(data, for: characteristic, type: CBCharacteristicWriteType.withResponse)
//                peripheral.readValue(for: characteristic)
//                print("Read finish")
//                peripheral.setNotifyValue(true, for: characteristic)
//                print("Notify Finish")
            }
            
//            let hexString: String = "0C11111111020000000000000000"
//            let commandData = NSData(data: hexString.data(using: .utf8)!)
//            let commandDataBase64 = commandData.base64EncodedString()
//            let commandDataBase64Data = Data(base64Encoded: commandDataBase64)!
//            peripheral.writeValue(commandDataBase64Data, for: characteristic, type: .withResponse)
//            peripheral.setNotifyValue(true, for: characteristic)
//
//            if characteristic.properties.contains(.notify) {
//                print("\(characteristic.uuid): properties contains .notify")
//                //let commandData = 0x0C 0x11111111 0x03 0000000000000064
//                let commandByteData = stringToBytes("0C11111111030000000000000000")!
//                let commandData = Data(bytes: commandByteData)
//                print(commandByteData)
//                peripheral.writeValue(commandData, for: characteristic, type: CBCharacteristicWriteType.withResponse)
//                //peripheral.setNotifyValue(true, for: characteristic)
//            }
            
            //peripheral.readValue(for: characteristic)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        switch characteristic.uuid {
        case imuCharacteristicCBUUID:
            print(characteristic.value ?? "no value")
        case writeCharacteristicCBUUID:
            print(characteristic.value ?? "no value")
        default:
            print("Unhandled Characteristic UUID: \(characteristic.uuid)")
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        print("ERROR PRINT ~~~~~~~~~~~~~~~~~~")
        print(characteristic)
        print(error)
    }
}


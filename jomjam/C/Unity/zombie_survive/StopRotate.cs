using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StopRotate : MonoBehaviour
{

    
    void Start()
    {
        transform.localPosition = new Vector3(0,0,0);
        transform.rotation = Quaternion.Euler(new Vector3(transform.parent.gameObject.transform.rotation.x, 0, transform.parent.gameObject.transform.rotation.z));
    }

    
    void Update()
    {
        transform.localPosition = new Vector3(0, 0, 0);
        transform.rotation = Quaternion.Euler(new Vector3(transform.parent.gameObject.transform.rotation.x, 0, transform.parent.gameObject.transform.rotation.z));
    }
}

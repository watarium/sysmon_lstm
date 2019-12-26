from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('dlls_model.h5', compile=False)
with open('dlls_token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('dlls_encode.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

max_len = int(model.input.shape[1])

testdata = 'ntdll.dll sechost.dll msvcrt.dll oleaut32.dll rpcrt4.dll ole32.dll advapi32.dll ws2_32.dll setupapi.dll nsi.dll gdi32.dll usp10.dll kernel32.dll user32.dll lpk.dll imm32.dll msctf.dll cfgmgr32.dll devobj.dll KernelBase.dll RpcRtRemote.dll cryptbase.dll sspicli.dll secur32.dll cryptsp.dll dnsapi.dll credssp.dll powrprof.dll slc.dll clusapi.dll winnsi.dll cryptdll.dll mswsock.dll wship6.dll rasadhlp.dll localspl.dll srvcli.dll spoolss.dll PrintIsolationProxy.dll tcpmon.dll snmpapi.dll wsnmp32.dll msxml6.dll shlwapi.dll clbcatq.dll wintrust.dll crypt32.dll msasn1.dll usbmon.dll WlS0WndH.dll WSDMon.dll WSDApi.dll webservices.dll FirewallAPI.dll version.dll fundisc.dll atl.dll fdPnp.dll gpapi.dll dsrole.dll winprint.dll userenv.dll profapi.dll netutils.dll cryptsp.dll rsaenh.dll win32spl.dll devrtl.dll SPInf.dll winsta.dll cscapi.dll wtsapi32.dll ntmarta.dll Wldap32.dll mxdwdrv.dll mxdwdrv.dll unidrvui.dll mxdwdui.dll comctl32.dll mxdwdui.dll ntprint.dll mscms.dll shell32.dll mxdwdui.dll ntprint.dll mscms.dll shell32.dll spfileq.dll cabinet.dll sfc.dll sfc_os.dll wkscli.dll netapi32.dll tapi32.dll ntprint.dll mscms.dll ntprint.dll mscms.dll mxdwdui.dll mxdwdui.dll mxdwdui.dll'
testdata = [testdata]

testdata_mat = tokenizer.texts_to_sequences(testdata)
data_sec = pad_sequences(testdata_mat, maxlen=max_len)
prediction = model.predict(np.array(data_sec))

predicted_label = encoder.classes_[np.argmax(prediction)]
print(predicted_label)
print('\nProbability')
for i in range(len(prediction[0])):
    print(str(encoder.classes_[i]) + ' : ' + str(prediction[0][i]))
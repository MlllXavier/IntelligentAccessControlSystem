import serial

portx = 'COM1'
bps = 115200
timex = 5

srl = serial.Serial(portx, bps, timeout=timex)


def Open_door():
    hex_str = 'dd 08 24 00 09'
    hex_stra = bytes.fromhex(hex_str)
    srl.write(hex_stra)


def Close_door():
    hex_str = 'dd 08 24 00 0A'
    hex_stra = bytes.fromhex(hex_str)
    srl.write(hex_stra)


def Warning_door():
    hex_str = 'dd 08 24 00 02'
    hex_stra = bytes.fromhex(hex_str)
    srl.write(hex_stra)


def UNwarning_door():
    hex_str = 'dd 08 24 00 03'
    hex_stra = bytes.fromhex(hex_str)
    srl.write(hex_stra)

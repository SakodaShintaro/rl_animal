import os
import shutil
import struct

import dnfile

from rl_animal_torch.config import ENV_PATH

# The v4 player writes one CSV row per step into a queue that a background thread drains,
# and at a high timescale the queue grows faster than it is written: a 24-actor run put the
# host out of memory after six hours. There is no switch for it, TrainingAgent adds the
# component unconditionally, so the entry point of the logging method is overwritten with
# `ret`. The rest of its IL stays in place, decodable but never reached. The original
# assembly is kept next to it as .orig.
ASSEMBLY = "animalAI_Data/Managed/Scripts.dll"
TYPE_NAME = "CSVWriter"
METHOD_NAME = "LogToCSV"
RET = 0x2A


def body_offset(path):
    pe = dnfile.dnPE(path)
    for row in pe.net.mdtables.TypeDef.rows:
        if str(row.TypeName) != TYPE_NAME:
            continue
        for entry in row.MethodList:
            if str(entry.row.Name) != METHOD_NAME:
                continue
            header = pe.get_offset_from_rva(entry.row.Rva)
            first = pe.__data__[header]
            if first & 0x3 == 0x2:
                return header + 1, first >> 2
            return header + 12, struct.unpack_from("<I", pe.__data__, header + 4)[0]

    raise AssertionError(f"{TYPE_NAME}.{METHOD_NAME} not found in {path}")


def main():
    path = os.path.join(os.path.dirname(ENV_PATH), ASSEMBLY)
    assert os.path.exists(path), f"{path} does not exist"

    offset, size = body_offset(path)
    data = bytearray(open(path, "rb").read())
    if data[offset] == RET:
        print(f"{TYPE_NAME}.{METHOD_NAME} already returns immediately")
        return

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print(f"backed up {backup}")
    data[offset] = RET
    open(path, "wb").write(data)
    print(f"{TYPE_NAME}.{METHOD_NAME}: {size} bytes of IL at {offset} now start with ret")


if __name__ == "__main__":
    main()

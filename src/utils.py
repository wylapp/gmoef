import jsonlines
from os import path
import time


def dump_final_result(pairs:list[tuple[int, int]])->str:
    dump_res = []
    for res in pairs:
        output_res = {"id": res[1], "label": res[0]}
        dump_res.append(output_res)
    filename = path.join("output", f"{str(int(time.time()))}-prediction.jsonl")
    with jsonlines.open(filename, mode='w') as writer:
        writer.write_all(dump_res)

    return filename

if __name__ == '__main__':
    dump_final_result([(2, 3), (3, 4), (4, 5)])
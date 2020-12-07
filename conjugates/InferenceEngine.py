import copy

class Engine():
    def __init__(self):
        print("Initiate ",self)

    def inference(self, pdf, data):
        pdfs = []
        size = len(data)
        if (size > 4):
            window = int(size / 4)
        else:
            window = size
        batches = [batch for batch in range(0, size, window)]
        if (len(batches) == 1):
            pdfs.append(pdf.update(data))
        else:
            for i in range(1, len(batches)):
                curr = batches[i]
                prev = batches[i - 1]
                batch = data[prev:curr]
                pdf=pdf.update(batch)
                pdfs.append(copy.copy(pdf))
        return pdfs
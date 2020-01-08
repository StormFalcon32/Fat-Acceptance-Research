import Preprocess as preprocess
import Ngrams as ngrams
import LDA as lda
import Visualize as visualize

if __name__ == '__main__':
    start = 2
    end = 30
    increment = 2
    runs = [1, 1, 0, 0]
    if runs[0]:
        preprocess.main(True)
    if runs[1]:
        ngrams.main()
    if runs[2]:
        lda.main(start, end, increment)
    if runs[3]:
        visualize.main(start, end, increment)

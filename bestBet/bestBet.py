import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets
import matplotlib.lines as lines
import numpy as np
import sys

delta = 1
maxBet = 100
odds = [1.95, 3.5, 4]

class BetText():
    def __init__(self, axes):
        self.betText = axes.annotate('wedge', xy=(2., -1),  xycoords='data',
            xytext=(35, 0), textcoords='offset points',
            size=10, va="center", visible=False,
            bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
            arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                            fc=(1.0, 0.7, 0.7), ec="none", 
                            patchA=None,
                            patchB=patches.Ellipse((2, -1), 0.5, 0.5),
                            relpos=(0.2, 0.5),
                            )
            )
        self.betPosition = axes.add_line(lines.Line2D([0], [0], marker='o', visible=False))
        self.bets = [0, 0, 0]

    def updateValues(self, bets, xy = None):
        for i in xrange(0, 3):
            self.bets[i] = self.bets[i] if bets[i] is None else bets[i]
        betTotal = sum(self.bets)
        self.betText.set_text('Bet: %.2f  %.2f  %.2f\nTeam 1 wins: %.2f$\nDraw:    %.2f$\nTeam 2 wins: %.2f$' % (self.bets[0], self.bets[1], self.bets[2], self.bets[0]*odds[0] - betTotal, self.bets[1]*odds[1] - betTotal, self.bets[2]*odds[2] - betTotal))
        if xy is not None:
            self.betText.xy = xy
            self.betPosition.set_xdata([xy[0]])
            self.betPosition.set_ydata([xy[1]])
            self.betText.set_visible(True)
            self.betPosition.set_visible(True)

class BetLines():
    def __init__(self, axes):
        self.lineColors = ['red', 'blue', 'brown']
        self.betLines = [None, None, None]
        self.labels = ['Team 1 wins', 'Draw', 'Team 2 wins']
        for i in xrange(0, 3):
            self.betLines[i] = \
                (lines.Line2D([], [], color=self.lineColors[i], label=self.labels[i]), \
                 lines.Line2D([], [], color=self.lineColors[i], ls='--'))
            axes.add_line(self.betLines[i][0])
            axes.add_line(self.betLines[i][1])

    def mainLines(self):
        return tuple(l1 for (l1, l2) in self.betLines)

    def updateLine(self, borderLines, xCoeff, yCoeff, c):
        f = lambda x: (x*xCoeff + c)/(-yCoeff)
        x = np.arange(0, maxBet, 0.1)
        y = f(x)
        sign = 1 if xCoeff*x[1] + yCoeff*(y[1] + 1) + c > 0 else -1
        y2 = y + sign*delta
        borderLines[0].set_xdata(x)
        borderLines[0].set_ydata(y)
        borderLines[1].set_xdata(x)
        borderLines[1].set_ydata(y2)

    def updateLines(self, fig, bet1Val):
        self.updateLine(self.betLines[0], -1, -1, bet1Val*(odds[0]-1))
        self.updateLine(self.betLines[1], odds[1] - 1, -1, -bet1Val)
        self.updateLine(self.betLines[2], -1, odds[2] - 1, -bet1Val)

def bestBet():
    '''
     Find "optimal" bet for a soccer game (if there is any)
    '''
    plt.ion()
    figBet = plt.figure()
    axPlot = figBet.add_subplot(111)
    figBet.subplots_adjust(bottom = 0.25)
    axSlider = figBet.add_subplot(919, axisbg='lightgoldenrodyellow', adjustable='datalim')

    axPlot.set_title('Bet for [%0.2f, %0.2f, %0.2f]' % tuple(odds))
    axPlot.set_xlim(0, maxBet)
    axPlot.set_ylim(0, maxBet)
    axPlot.set_xlabel('Bet for X (draw)')
    axPlot.set_ylabel('Bet for team 2')

    # fancy stuffs    
    betText = BetText(axPlot)
    betLines = BetLines(axPlot)

    axSlider.set_position([0.25, 0.1, 0.65, 0.03])
    sliderTeam1 = widgets.Slider(axSlider, label='Bet for team 1', valmin=0, \
                                valmax=maxBet, valinit=1)
   
    # call-back functions
    def slider_onChanged(bet1Val):
        betLines.updateLines(figBet, bet1Val)
        betText.updateValues([bet1Val, None, None])
        figBet.canvas.draw()

    def figure_onMousePress(event):
        if event.inaxes != axPlot:
            return
        betText.updateValues([sliderTeam1.val, event.xdata, event.ydata], (event.xdata, event.ydata))
        figBet.canvas.draw()

    figBet.canvas.mpl_connect('button_press_event', figure_onMousePress)
    sliderTeam1.on_changed(slider_onChanged)

    betLines.updateLines(figBet, sliderTeam1.val)
    #plt.legend(betLines.mainLines(), betLines.labels, loc='upper right')
    axPlot.legend()
    plt.show()
    raw_input()


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        odds = [float(sys.argv[i]) for i in xrange(1, 4)]
    if len(sys.argv) >= 5:
        maxBet = float(sys.argv[4])
        delta = maxBet / 100.
    bestBet()
    
    

'''
'''

class visualization:


    def plot_posterior(dset=0, h=0.01, enable=[1,1,1,1], cc=False, savefile=True):

        # makes folder if not exists
        if not os.path.isdir(FILEPATH):
            os.makedirs(FILEPATH)

        dsetnum = len(data)

        X1, X2, y = [],[],[]

        for i in range(dsetnum):
            if cc:
                X, tempy, _, _ = select_dataset(i, cc=True)
            else:
                X, tempy, _, _ = select_dataset(i)
            y.append(tempy)
            X1.append(X[:,0])
            X2.append(X[:,1])
            # X1[i], X2[i] = , X[:,1]

        premax = round(X.max())
        postmax = round(uX0.max())

        prerng = (-premax,premax)
        postrng = (-postmax,postmax)
        clfname = mods.copy()

        md = 3 # model to be plotted as figure 3; [0,5]
        fsize = 15 # figure fontsize

        # cmap = 'RdBu_r'

        if enable[0] == 1:

            ################ TYPE 1 ################

            flen = dsetnum# figure size

            # fig, axs = plt.subplots(1,flen, figsize=(30,9))
            fig = plt.figure(figsize=(30,5))

            for i in range(dsetnum):
                axs = fig.add_subplot(1,flen,i+1)
                pcm = axs.scatter(X1[i], X2[i], c=get_colors(colors, y[i]), s=20)
                axs.add_patch(Rectangle(
                                (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))

                axs.set_title(data[i], fontsize=fsize)
                axs.set_xticks([])
                axs.set_yticks([])

                axs.set_xlim(np.multiply(postrng, 1))
                axs.set_ylim(np.multiply(postrng, 1)) 

            # plt.suptitle('Simulated Datasets', fontsize=25) 

            if savefile:
                plt.savefig('figs/' + str(datetime.today())[:10] + '_simulated_datasets_' + '(' + str(h) + ').png')

            fig = plt.figure(figsize=(30,4))

            for i in range(dsetnum):
                axs = fig.add_subplot(1,flen,i+1)
                pcm = axs.scatter(xx, yy, c=true_zz[i], cmap=cmap, s=5)
                axs.add_patch(Rectangle(
                                (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))
                # fig.colorbar(pcm, ax=axs)  

                axs.set_title(data[i], fontsize=fsize)
                axs.set_xticks([])
                axs.set_yticks([])
                axs.set_xlim(np.multiply(postrng, 1))
                axs.set_ylim(np.multiply(postrng, 1)) 

            # plt.suptitle('True Posteriors', fontsize=25) 

            if savefile:
                plt.savefig('figs/' + str(datetime.today())[:10] + '_true_posteriors_' + '(' + str(h) + ').png')

            ################ END OF TYPE 1 ################

        if enable[1] == 1:

            ################ TYPE 2 ################

            # fig = plt.figure(figsize=(13,16))
            cnt = 0
            cnt2 = -1
            cnt3 = -1
            fig = plt.figure(figsize=(30,4*(dsetnum)))

            for j in range(dsetnum):
                # fig = plt.figure(figsize=(30,4))
                for i in range(len(post[0])+2):
                    
                    # i += 1
                    # print(i)
                    cnt += 1
                    axs = fig.add_subplot(dsetnum,9,cnt)

                    if cnt % 9 == 1:
                        cnt3 += 1
                        pcm = axs.scatter(X1[cnt3], X2[cnt3], c=get_colors(colors, y[cnt3]), s=10)
                    elif cnt % 9 == 2:
                        cnt2 += 1
                        pcm = axs.scatter(xx, yy, c=true_zz[cnt2], cmap=cmap, s=5)
                    else:
                        pcm = axs.scatter(uX0, uX1, c=post[j][i-2], cmap=cmap, s=5)

                    axs.add_patch(Rectangle(
                                    (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))
                    # fig.colorbar(pcm, ax=axs)
                    axs.set_xticks([])
                    axs.set_yticks([])

                    if i == 0:
                        axs.set_ylabel(data[j], fontsize=18)
                    if j == 0 and i > 1:
                        axs.set_title(clfname[i-2], fontsize=fsize) 
                    elif j == 0 and i == 0:
                        axs.set_title('Simulation Data', fontsize=fsize) 
                    elif j == 0 and i == 1:
                        axs.set_title('True Posterior', fontsize=fsize) 

                    # axs.set_xlim(np.multiply(postrng, 1))
                    # axs.set_ylim(np.multiply(postrng, 1))  

            if savefile:
                plt.savefig('figs/' + str(datetime.today())[:10] + '_estimated_posterior_' + '(' + str(h) + ').png')

            ################ END OF TYPE 2 ################

        if enable[2] == 1:

            ################ TYPE 3 ################

            fig, axs = plt.subplots(2,2, figsize=(8,8))

            axs[0,0].set_title('Gaussian XOR', fontsize=fsize) 
            axs[0,0].set_xticks([])
            axs[0,0].set_yticks([])
            axs[0,0].set_xlim(np.multiply(postrng, 1))
            axs[0,0].set_ylim(np.multiply(postrng, 1)) 
            axs[0,0].scatter(X1[0], X2[0], c=get_colors(colors, y[0]), s=20)
            axs[0,0].add_patch(Rectangle(
                            (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))

            axs[0,1].set_title('True Posterior', fontsize=fsize) 
            axs[0,1].set_xticks([])
            axs[0,1].set_yticks([])
            axs[0,1].set_xlim(np.multiply(postrng, 1))
            axs[0,1].set_ylim(np.multiply(postrng, 1)) 
            axs[0,1].scatter(xx, yy, c=true_zz[0], cmap=cmap, s=1)
            axs[0,1].add_patch(Rectangle(
                            (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))

            axs[1,0].set_title('RF', fontsize=fsize) 
            axs[1,0].set_xticks([])
            axs[1,0].set_yticks([])
            axs[1,0].set_xlim(np.multiply(postrng, 1))
            axs[1,0].set_ylim(np.multiply(postrng, 1)) 
            axs[1,0].scatter(uX0, uX1, c=post[0][5], cmap=cmap, s=5)
            axs[1,0].add_patch(Rectangle(
                            (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))

            axs[1,1].set_title('DN', fontsize=fsize) 
            axs[1,1].set_xticks([])
            axs[1,1].set_yticks([])
            axs[1,1].set_xlim(np.multiply(postrng, 1))
            axs[1,1].set_ylim(np.multiply(postrng, 1)) 
            axs[1,1].scatter(uX0, uX1, c=post[0][4], cmap=cmap, s=5)
            axs[1,1].add_patch(Rectangle(
                            (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))

            if savefile:
                plt.savefig('updated_figure.png')
        

            ################ END OF TYPE 3 ################        
        
        if enable[3] == 1:

            ################ TYPE 4 ################

                row = 2
                col = 1+3#+1
                
                # plt.suptitle('Mean Hellinger Distance', fontsize=25)

                # dsetnum = 1

                for j in range(dsetnum):

                    cnt = 0
                    
                    # fig = plt.figure()           
                    fig = plt.figure(figsize=(4*col,4*row))

                    ax = fig.add_subplot(111)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax.spines['top'].set_color('none')
                    ax.spines['bottom'].set_color('none')
                    ax.spines['left'].set_color('none')
                    ax.spines['right'].set_color('none')

                    ax.set_ylabel(data[j], fontsize=18)

                    axlist = []
                    
                    for k in range(row):
                        for i in range(col):
                            cnt += 1
                            # if k == 0 and i == 0:
                            #     axs = fig.add_subplot(row,col,cnt)
                            #     # fig.ylabel(data[j], fontsize=18)
                            # else:
                            axs = fig.add_subplot(row,col,cnt)#, sharey=axs)
                            axlist.append(axs)

                            axs.set_xticks([])
                            axs.set_yticks([])

                            # simulation dataset
                            if cnt == 1:
                                pcm = axs.scatter(X1[j], X2[j], c=y[j], cmap=cmap, s=10) #c=get_colors(colors, y[j]), 
                                if cc:
                                    axs.add_patch(Circle((0,0), radius=1, linewidth=2, edgecolor='k', fill=False))
                                else:
                                    axs.add_patch(Rectangle(
                                                (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))                                            
                            # true posterior
                            elif cnt == col+1:  
                                # if j == 2:
                                #     a, b, c, d, e, f, g = pdf_spiral(500000, noise=0.9, K=2)
                                #     pcm = axs.scatter(a[:,0],a[:,1], c=c, cmap='PRGn_r', s=0.1)       
                                #     if cc:
                                #         axs.add_patch(Circle((0,0), radius=1, linewidth=2, edgecolor='k', fill=False))                      
                                #     else:
                                #         axs.add_patch(Rectangle(
                                #                     (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))                               
                                # else:    
                                if cc:
                                    pcm = axs.scatter(xx, yy, c=Ctrue_zz[j], cmap=cmap, s=1)
                                    axs.add_patch(Circle((0,0), radius=1, linewidth=2, edgecolor='k', fill=False))
                                else:
                                    pcm = axs.scatter(xx, yy, c=true_zz[j], cmap=cmap, s=1)
                                    axs.add_patch(Rectangle(
                                                (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))
                            # estimated posterior
                            elif cnt > 1 and cnt < col+1:
                                if cc:
                                    pcm = axs.scatter(uX0, uX1, c=Cpost[j][i-1], cmap=cmap, s=1)
                                    axs.add_patch(Circle((0,0), radius=1, linewidth=2, edgecolor='k', fill=False))
                                else:                    
                                    pcm = axs.scatter(uX0, uX1, c=post[j][i-1], cmap=cmap, s=1)
                                    axs.add_patch(Rectangle(
                                                    (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False))
                            # colorbar
                            # elif i == col-1:
                            #     # fig2 = pl.figure(figsize=(0.5, 4))
                            #     axs = plt.imshow([[0,1]], cmap='RdBu_r')
                            #     plt.gca().set_visible(False)
                            #     cax = plt.axes([0.1, 0.2, 0.8, 0.6])
                            #     plt.colorbar(orientation="vertical", cax=cax)

                            # hellinger
                            else:     
                                if cc:
                                    pcm = axs.scatter(CuX_outside[:,0], CuX_outside[:,1], c=COUT_hellinger[j][i-1], cmap='binary', s=5)
                                    axs.add_patch(Circle((0,0), radius=1, linewidth=2, edgecolor='k', fill=False, hatch='/'))
                                    temptitle = 'Mean Distance:' + str(round(COUT_hellinger[j][i-1].mean(),2))
                                    axs.set_title(temptitle, fontsize=fsize)
                                else:
                                    pcm = axs.scatter(uX_outside[:,0], uX_outside[:,1], c=OUT_hellinger[j][i-1], cmap='binary', s=5)
                                    axs.add_patch(Rectangle(
                                                    (-1,-1), 2, 2, linewidth=2, edgecolor='k', fill=False, hatch='/'))
                                    temptitle = 'Mean Distance:' + str(round(OUT_hellinger[j][i-1].mean(),2))
                                    axs.set_title(temptitle, fontsize=fsize)
                            # fig.colorbar(pcm, ax=axs)  
                            if k == 0 and i == 0:                            
                                axs.set_title('Simulation Data', fontsize=fsize)
                                axs.set_xticks([])
                                axs.set_yticks([])

                            if k == 1 and i == 0:                    
                                axs.set_title('True Posterior', fontsize=fsize)
                                axs.set_xticks([])
                                axs.set_yticks([])

                            # if k == 1 and i > 0:
                            #     templabel = 'Mean Hellinger Distance' + clfname[i-1]
                            #     axs.set_title(templabel, fontsize=18)

                            if k == 0 and i == 1:
                                axs.set_ylabel('Estimated Posterior', fontsize=18)
                                axs.set_yticks([])

                            if k == 1 and i == 1:
                                axs.set_ylabel('Point-wise Hellinger', fontsize=18)
                                axs.set_yticks([])

                            if k == 0 and i > 0 and i < col:
                                axs.set_title(clfname[i-1], fontsize=fsize)

                            # if i == col-1:
                            #     pass
                            # else:
                            axs.set_xlim(np.multiply(postrng, 1))
                            axs.set_ylim(np.multiply(postrng, 1))  
                        # cbar = fig.colorbar(pcm, ax=axlist,shrink=0.5)
                        # cbar.set_ticks([0,1])

                    if savefile:
                        if cc:
                            path = 'figs/' + str(datetime.today())[:10] + '_' + data[j] + '_Cfigures.png'# + '(' + str(h) + ')'
                        else:
                            path = 'figs/' + str(datetime.today())[:10] + '_' + data[j] + '_figures.png'# + '(' + str(h) + ')'
                        plt.tight_layout(pad=2)
                        plt.savefig(path, bbox_inches='tight')

                    plt.show()               

                # plt.subplots(2,2)


            ################ END OF TYPE 4 ################  

            ################ TYPE 5 ################
        if enable[4] == 1:
            
            row = 2
            col = 4
            interp_method = 'cubic' #interpolation methods
            

            # plt.suptitle('Mean Hellinger Distance', fontsize=25)

            # dsetnum = 1

            for ii, j in enumerate([2,4]):

                cnt = 0

                widths = [4,4,4,8]
                fig = plt.figure(figsize=(4*(col+1),4*row))
                spec = fig.add_gridspec(ncols=col, nrows=row, width_ratios=widths)

                ax = fig.add_subplot(111)
                ax.set_xticks([])
                ax.set_yticks([])

                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')

                ax.set_ylabel(data[j], fontsize=18)

                # axlist = []
                
                for k in range(row):
                    for i in range(col):
                        cnt += 1
                        
                        axs = fig.add_subplot(spec[k,i])#, sharey=axs)
                        # axs = fig.add_subplot(row,col,cnt)#, sharey=axs)
                        # axlist.append(axs)

                        # simulation dataset
                        if cnt == 1:
                            pcm = axs.scatter(X1[j], X2[j], c=y[j], cmap=cmap, s=10) #c=get_colors(colors, y[j]), 

                        # true posterior
                        elif cnt == col+1:     
                            pcm = axs.scatter(xx, yy, c=Ctrue_zz[j], cmap=cmap, s=1)

                        # estimated posterior
                        elif cnt > 5 and cnt < 8:
                            if cnt == 7:
                                if j == 2:
                                    # pcm = axs.scatter(MT_4[:,3],MT_4[:,5], c=MT_4[:,0], cmap=cmap, s=10)
                                    grid_near = griddata((MT_4[:,3], MT_4[:,5]), MT_4[:,0], (uX0,uX1), method=interp_method, rescale=True) #, fill_value=0.5
                                    pcm = axs.scatter(uX0,uX1, c=grid_near, cmap=cmap, s=1)
                                elif j == 4:
                                    # pcm = axs.scatter(MT_2[:,3],MT_2[:,5], c=MT_2[:,0], cmap=cmap, s=10)                              
                                    grid_near = griddata((MT_2[:,3], MT_2[:,5]), MT_2[:,0], (uX0,uX1), method=interp_method)
                                    pcm = axs.scatter(uX0,uX1, c=grid_near, cmap=cmap, s=1)
                                axs.set_title('HUMAN', fontsize=fsize)                            

                            else:
                                pcm = axs.scatter(uX0, uX1, c=Cpost[j][2], cmap=cmap, s=1)
                                axs.set_title(mods[2], fontsize=fsize)

                        elif cnt > 1 and cnt < 4:
                            pcm = axs.scatter(uX0, uX1, c=Cpost[j][i-1], cmap=cmap, s=1)
                            axs.set_title(mods[i-1], fontsize=fsize)

                        if cnt != 4 and cnt != 8:
                            circle = Circle((0, 0), 3, linewidth=1, edgecolor='k', facecolor='none') #outer bounding circle
                            axs.add_patch(Circle((0,0), radius=1, linewidth=2, ls='--',edgecolor='r', fill=False))
                            axs.add_patch(circle)
                            pcm.set_clip_path(circle)

                        if k == 0 and i == 0:                            
                            axs.set_title('Simulation Data', fontsize=fsize)

                        if k == 1 and i == 0:                    
                            axs.set_title('True Posterior', fontsize=fsize)

                        if i == 1 and (k == 0 or k == 1):
                            axs.set_ylabel('Estimated Posterior', fontsize=18)

                        if i < 3:
                            axs.set_xticks([])
                            axs.set_yticks([])

                        # axs.add_patch(circle)
                        # pcm.set_clip_path(circle)

                        ###################
                        lines = []

                        if cnt == 4:
                            for m in range(len(new_mods)-1):
                                l1, = axs.plot(rad, pALL_rad[j][m]) # whole range
                                lines.append(l1)                        
                            # l2, = axs.plot(new_rad[-ii+1], human_pALL_rad[-ii+1]) # different radius range
                            l2 = sns.lineplot(human_i[-ii+1], human_p[-ii+1], ci=95) # different radius range
                            lines.append(l2)
                            axs.legend(lines, new_mods, prop={'size':9})
                            axs.set_title('Class 1 Posterior', fontsize=fsize)
                        elif cnt == 8:
                            for m in range(len(new_mods)-1):
                                l1, = axs.plot(rad, hALL_rad[j][m]) # whole range
                                lines.append(l1)                        
                            # l2, = axs.plot(new_rad[-ii+1], human_hALL_rad[-ii+1]) # different radius range
                            l2 = sns.lineplot(human_i[-ii+1], human_h[-ii+1], ci=95) # different radius range
                            lines.append(l2)
                            axs.legend(lines, new_mods, prop={'size':9})
                            axs.set_title('Hellinger Distance', fontsize=fsize)
                        elif cnt == 7:
                            axs.set_xlim([-3,3])
                            axs.set_ylim([-3,3])
                        else:
                            axs.set_xlim(np.multiply(postrng, 1))
                            axs.set_ylim(np.multiply(postrng, 1))
                        

    ###################
                
                # ax2 = plt.subplot(248)
                # # axs = fig.add_subplot(spec[2,4])

                # # [pALL_rad, hALL_rad]
                # # [human_pALL_rad, human_hALL_rad]
                # for m in range(len(new_mods)-1):
                #     ax2.plot(rad, pALL_rad[2][m]) # whole range
                #     ax2.plot(new_rad[0], human_pALL_rad[0]) # different radius range
                #     ax2.set_xlim([0,3])
                #     ax2.legend(new_mods, prop={'size':9})            

                if savefile:
                    if cc:
                        path = 'figs/' + str(datetime.today())[:10] + '_' + data[j] + '_Cfigures_human.png'# + '(' + str(h) + ')'
                    else:
                        path = 'figs/' + str(datetime.today())[:10] + '_' + data[j] + '_figures_human.png'# + '(' + str(h) + ')'
                    plt.tight_layout(pad=2)
                    plt.savefig(path, bbox_inches='tight')

                plt.show()   
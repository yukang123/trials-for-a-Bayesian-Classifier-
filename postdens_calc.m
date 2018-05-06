function dens=postdens_calc(x,mu,cova,p)
inv_co=inv(cova);
dens=-(x-mu)*inv_co*(x-mu)'/2-log(det(cova))/2+log(p);
end
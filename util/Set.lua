function addToSet(set, key)
  if key~=nil then
    set[key]=true
  end
end
function removeFromSet(set, key)
  if key~=nil then
    set[key]=nil
  end
end
function setContains(set, key)
    return set[key]~=nil
end

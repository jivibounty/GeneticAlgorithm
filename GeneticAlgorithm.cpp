#include "GeneticAlgorithm.hpp"

#include <cstdlib>
#include <utility>
#include <iostream>

namespace hg
{
   
     static const int RAND_MAX_VALUE = 10000000;
    
     GeneticAlgorithm::GeneticAlgorithm(const std::vector<GAValue>& initialValues,
                                size_t populationSize,
                                size_t epochs,
                                size_t numParents,
                                size_t numGenesToModify,
                                void* userData,
                                std::function<double(GAValue*, size_t, void*)> fitnessFunction)
    {
        mEpochs = epochs;
        mNumParents = numParents;
        mUserData = userData;
        mFitnessFunction = fitnessFunction;
        mValueCount = initialValues.size();
        mNumGenesToModify = std::min<size_t>(std::max<size_t>(1, numGenesToModify), mValueCount);
        
        mSolutions.clear();
        for (size_t i = 0; i < populationSize; ++i)
        {
           mSolutions.insert(mSolutions.end(), initialValues.begin(), initialValues.end());
        }
        
        for (auto solution : mSolutions)
        {
            mutate(solution);
        }
        mFitnessData.resize(populationSize);
        evaluateFitnessFunction();
    }
    
    GeneticAlgorithm::~GeneticAlgorithm()
    {
    }
   
    static size_t pickRandomParent(size_t parentCount)
    {
        size_t parentIndex = static_cast<size_t>(rand() % static_cast<int>(parentCount));
       
        return parentIndex;
    }
   
    static size_t getGeneIndex(size_t populationIndex, size_t geneCount, size_t geneOffset)
    {
        size_t geneIndex = populationIndex * geneCount + geneOffset;
        
        return geneIndex;
    }
   
    static size_t pickRandomGene(size_t populationIndex, size_t geneCount)
    {
        size_t geneOffset = static_cast<size_t>(rand() % static_cast<int>(geneCount));
        size_t geneIndex = getGeneIndex(populationIndex, geneCount, geneOffset);
        
        return geneIndex;
    }
        
    bool GeneticAlgorithm::advance()
    {
        ++mCurrentEpoch;
       
        for (size_t i = 0; i < mNumGenesToModify; ++i)
        {
            update();
        }
       
        if (mCurrentEpoch >= mEpochs)
        {
            return false;
        }
        return true;
    }
        
    int GeneticAlgorithm::getProgress() const
    {
        return static_cast<int>(0.1 + 100.0 * (double)mCurrentEpoch / (double)mEpochs);
    }
        
    bool GeneticAlgorithm::hasSolution() const
    {
        return mFitnessData[0].fitness >= 0;
    }
   
    double GeneticAlgorithm::getBestScore() const
    {
        return mFitnessData[0].fitness;
    }
        
    const std::vector<GAValue>& GeneticAlgorithm::getBestSolution()
    {
        size_t index = mFitnessData[0].index * mValueCount;
        mBestSolution.clear();
        mBestSolution.insert(mBestSolution.end(), mSolutions.begin() + index, mSolutions.begin() + index + mValueCount);
        
        return mBestSolution;
    }
    
    void GeneticAlgorithm::update()
    {
        size_t offset = mNumParents;
        size_t len = mFitnessData.size() - offset;
        size_t lenDelta = len / 3;
        size_t end = lenDelta;
        size_t geneCount = mValueCount;
        
        // mutate
        for (size_t i = offset; i < end; ++i)
        {
            size_t parent = mFitnessData[pickRandomParent(mNumParents)].index;
            size_t child = mFitnessData[i].index;
            copyGenes(parent, child);
           
            size_t geneIndex = pickRandomGene(child, geneCount);
            mutate(mSolutions[geneIndex]);
        }
        
        offset = end;
        end = offset + lenDelta;
        
        // crossover
        offset = offset - (offset % 2);
        end = end - (end % 2);
        for (size_t i = offset; i < end; i += 2)
        {
            size_t parent = mFitnessData[pickRandomParent(mNumParents)].index;
            size_t child = mFitnessData[i].index;
            size_t child2 = mFitnessData[i + 1].index;
            copyGenes(parent, child);
            copyGenes(parent, child2);
           
            size_t geneIndex = pickRandomGene(child, geneCount);
            size_t gene2Index = getGeneIndex(child2, geneCount, geneIndex % geneCount);
            crossOver(mSolutions[geneIndex], mSolutions[gene2Index]);
        }
        
        offset = end;
        end = len;
        
        // translate
        for (size_t i = offset; i < end; ++i)
        {
            size_t parent = mFitnessData[pickRandomParent(mNumParents)].index;
            size_t child = mFitnessData[i].index;
            copyGenes(parent, child);
           
            size_t geneIndex = pickRandomGene(child, geneCount);
            translate(mSolutions[geneIndex]);
        }
        
        evaluateFitnessFunction();
    }
        
    void GeneticAlgorithm::mutate(GAValue& value)
    {
        switch(value.dataType)
        {
            case GADataType::Integer:
                value.numberValue.i = value.minValue.i + std::rand() % (value.maxValue.i - value.minValue.i + 1);
                break;
            case GADataType::Float:
                value.numberValue.f = value.minValue.f + (std::rand() % RAND_MAX_VALUE) / static_cast<float>(RAND_MAX_VALUE - 1) * (value.maxValue.f - value.minValue.f);
                break;
            case GADataType::Double:
                value.numberValue.d = value.minValue.d + (std::rand() % RAND_MAX_VALUE) / static_cast<float>(RAND_MAX_VALUE - 1) * (value.maxValue.d - value.minValue.d);
                break;
            default:
                break;
        }
        
        ensureLimits(value);
    }
        
    void GeneticAlgorithm::crossOver(GAValue& value1, GAValue& value2)
    {
        std::swap(value1.dataType, value2.dataType);
        std::swap(value1.numberValue, value2.numberValue);
    }
        
    void GeneticAlgorithm::translate(GAValue& value)
    {
        int diff = -1;
        if ((std::rand() % 2) == 0)
        {
            diff = 1;
        }
        
        switch(value.dataType)
        {
            case GADataType::Integer:
                value.numberValue.i += diff;
                break;
            case GADataType::Float:
                value.numberValue.f += static_cast<float>(diff);
                break;
            case GADataType::Double:
                value.numberValue.d += static_cast<double>(diff);
                break;
            default:
                break;
        }
        
        ensureLimits(value);
    }
        
    void GeneticAlgorithm::ensureLimits(GAValue& value)
    {
        switch(value.dataType)
        {
            case GADataType::Integer:
                value.numberValue.i = std::min(std::max(value.numberValue.i, value.minValue.i), value.maxValue.i);
                break;
            case GADataType::Float:
                value.numberValue.f = std::min(std::max(value.numberValue.f, value.minValue.f), value.maxValue.f);
                break;
            case GADataType::Double:
                value.numberValue.d = std::min(std::max(value.numberValue.d, value.minValue.d), value.maxValue.d);
                break;
            default:
                break;
        }
    }
        
    void GeneticAlgorithm::copyGenes(size_t parent, size_t child)
    {
        size_t geneCount = mValueCount;
        size_t parentOffset = parent * geneCount;
        size_t childOffset = child * geneCount;
       
        for (size_t i = 0; i < geneCount; ++i)
        {
            size_t parentIndex = parentOffset + i;
            size_t childIndex = parentIndex + i;
            mSolutions[childIndex] = mSolutions[parentIndex];
        }
    }
   
    // Custom comparison function (e.g., for descending order)
    bool compareDescending(FitnessData a, FitnessData b) {
        return a.fitness > b.fitness; 
    }
    
    void GeneticAlgorithm::evaluateFitnessFunction()
    {
        for (size_t i = 0; i < mFitnessData.size(); ++i)
        {
            mFitnessData[i].index = i;
            mFitnessData[i].fitness = mFitnessFunction(((GAValue*)mSolutions.data()) + (i * mValueCount), mValueCount, mUserData);
            //std::cout << mFitnessData[i].fitness << std::endl;
            //mFitnessData[i].fitness = mFitnessFunction(((GAValue*)mSolutions.data()) + (i * mValueCount), mValueCount, mUserData);
            //std::cout << mFitnessData[i].fitness << std::endl;
        }
      
        std::sort(mFitnessData.begin(), mFitnessData.end(), compareDescending);
        
        for (size_t i = 0; i < mFitnessData.size(); ++i)
        {
            std::cout << mFitnessData[i].index << ":" << mFitnessData[i].fitness << std::endl;
        }
   }
    
}
